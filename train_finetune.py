import argparse
import sys
import shutil
import tensorflow as tf
from TXT import*
from Model import Base_GPT
from Layers import ExpSchedule,LinearSchedule,load_weights,lm_loss,cl_loss,save_weights_lora,load_weights_lora
from sklearn.metrics import accuracy_score

import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_features', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='Files')
    parser.add_argument('--save_dir', type=str, default='Files/F_weights')
    parser.add_argument('--txt_dir', type=str, default='Files')
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--freeze_emb', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--gr_acc', type=int, default=8) #gradient accumulation parameter 
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    
    text_encoder = TextEncoder(data_dir + '/encoder_bpe_40000.json',data_dir + '/vocab_40000.bpe')
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    encoder['_start_'] = len(encoder)
    encoder['_end_'] = len(encoder)
    
    FAVOR = True
    def acc_(model,ds):
        c = 0
        counter = 0
        model.train = False
        for step_2, (x_batch_val, y_batch_val) in enumerate(ds):
            H = model(x_batch_val, training=False)
            c = accuracy_score(y_batch_val ,tf.cast(tf.math.argmax(tf.nn.softmax(H[2]), axis = 1),tf.int32))+c
            counter = counter +1
        model.train = True
        return str(c/(counter))
    
    l = []
    for path, subdirs, files in os.walk(txt_dir):
        for name in files:
            l.append(os.path.join(path, name))
            
    l = [a for a in l if ('.txt' in a[len(a)-4:])] 

    # text_encoder.encode(verbose = False)
    def dataset_fn(dataset):
      lt = [str(n.numpy().decode("utf-8")) for n in list(dataset.map(lambda x: x))]
      ds_list = text_encoder.encode(lt,verbose = True)
      ds_list = [word for word in ds_list if len(word) < 510]
      con = []
      ds_eq = []
      for i in ds_list:
          if len(con + i) > 510:
              con = [encoder['_start_']] + con + [encoder['_end_']]
              # con = con + [0]*(512-len(con))
              ds_eq.append(con)
              con = []
          else:
              con = con + i
      ds_eq = text_encoder.mask(ds_eq)
      # ds_list = tf.ragged.constant(ds_eq)
      return tf.data.Dataset.from_tensor_slices((ds_eq[0],ds_eq[1]))

    ds = tf.data.TextLineDataset(
        l,
        compression_type=None,
        buffer_size=4,
        num_parallel_reads=4,
    )
    
    print("tokenizing txt files...")
    
    train_ds  = ds.apply(dataset_fn)
    train_ds  = train_ds.batch(batch)
    card = train_ds.cardinality()
    
    # train_ds = tf.data.Dataset.load("/home/borz/Desktop/test_proj/Files/fine",element_spec=((tf.TensorSpec(shape=(512, 2), dtype=tf.int64, name=None), tf.TensorSpec(shape=(512), dtype=tf.float32, name=None))))
    # train_ds  = train_ds.batch(batch)
    model = Base_GPT(n_vocab = n_vocab,n_special = 0, n_ctx = n_ctx, 
                  n_embd = n_embd,train = True,freeze_emb = freeze_emb,
                  n_head = n_head,n_layer = n_layer, LoRA = False,FAVOR = FAVOR, random_features = random_features)

    learning_rate = LinearSchedule(warmup_steps = 20,decay = 2*card + 20,lr = 3.25e-5)
    
    optimizer= tf.keras.optimizers.Adam(
                                            learning_rate = learning_rate,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-08,
                                            weight_decay=0.01,
                                            clipvalue=1,
                                            global_clipnorm=1,
                                            jit_compile=True,
                                        )
    

    for i in train_ds.take(1):x = i

    h = model(x)
    model.summary()
    

    model = load_weights(model,n_ctx = n_ctx,n_special = 0 , n_embd = n_embd, freeze_emb = freeze_emb ,weights_shapes_path =  data_dir, weights_path = data_dir, names_path = data_dir)
    start = time.time()
  
    @tf.function
    def grad_step(x_batch_train,accum_gradient):

        with tf.GradientTape() as tape:

          H = model(x_batch_train, training=True)
          
          l1 = lm_loss(x_batch_train[0], H[0], H[1],n_ctx = n_ctx)
          train_loss = tf.reduce_mean(l1)
          
        grads = tape.gradient(train_loss , model.trainable_weights)
        
        accum_gradient = [(accum_gradient+grad) for accum_gradient, grad in zip(accum_gradient, grads)]

        return train_loss,accum_gradient
    
    @tf.function
    def apply_grad(accum_gradient,gr_acc):
        accum_gradient = [this_grad/gr_acc for this_grad in accum_gradient]
        optimizer.apply_gradients(zip(accum_gradient, model.trainable_weights))
        return
    
    acc = 0
    print("training...")
    for epoch in range(epochs):
        
        train_vars = model.trainable_variables
        accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
        l_cum = np.zeros(10)
        l1 = 0
  
        for step, x_batch_train in enumerate(train_ds):
            
            if acc < gr_acc:
                l1_,accum_gradient = grad_step(x_batch_train,accum_gradient)
                l1 = l1 + l1_
                acc = acc + 1
            else:
                apply_grad(accum_gradient,gr_acc)
                
                accum_gradient = [tf.zeros_like(this_var) for this_var in train_vars]
                 
                l1 = l1/gr_acc
                
                l_cum[0] = tf.reduce_mean(l1)
                l_cum = np.roll(l_cum, 1)
                
                l1 = 0
                
                l1_,accum_gradient = grad_step(x_batch_train,accum_gradient)
                l1 = l1 + l1_
                
                acc = 1
            

            sys.stdout.write('\r'+("epoch: " + str(epoch+1) + " step: " +str(step+1) + "/" + str(train_ds.cardinality().numpy()) + " loss: " + str(np.sum(l_cum)/25) + " memory: " + str(round(tf.config.experimental.get_memory_info('GPU:0')["peak"]*1e-6)) + "   "))
            
    end = time.time()
    print(end - start)
    print(round(tf.config.experimental.get_memory_info('GPU:0')["peak"]*1e-6) )
    c = 0
    
    names = {}
    
    for I,i in enumerate(model.layers):
        for J,j in enumerate(model.layers[I].weights):
            np.save(save_dir+str("/") +str(c)+".npy", model.layers[I].weights[J].numpy(), allow_pickle=False)
            names[str(c)] = model.layers[I].weights[J].name
            c = c+1
            
    import json

    with open(save_dir + str("/") + str("data.json"), 'w') as fp:
        json.dump(names, fp)
    
    shutil.copyfile(data_dir + '/names_emb.json', save_dir + '/names_emb.json')
    shutil.copyfile(data_dir + '/names_emb_f.json', save_dir + '/names_emb_f.json')
    shutil.copyfile(data_dir + '/params_shapes.json', save_dir + '/params_shapes.json')

    
    
    
