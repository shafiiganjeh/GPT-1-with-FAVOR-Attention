import argparse
import sys
import tensorflow as tf
from TXT import*
from Model import Finetune_tGPT
from Layers import ExpSchedule,LinearSchedule,load_weights,lm_loss,cl_loss
from sklearn.metrics import accuracy_score

import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--FAVOR', type=bool, default=True)
    parser.add_argument('--random_features', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='Files/')
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--freeze_emb', type=bool, default=True)
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--epochs', type=int, default=1)
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    
    text_encoder = TextEncoder(data_dir + 'encoder_bpe_40000.json',data_dir + 'vocab_40000.bpe')
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    
    
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
    
    train_ds = tf.data.Dataset.load(data_dir + "/fine")
    val = tf.data.Dataset.load(data_dir + "/fine").batch(batch)
    

        

    model = Finetune_tGPT(n_vocab = n_vocab,n_special = 0, n_ctx = n_ctx, 
                  n_embd = n_embd,train = True,freeze_emb = freeze_emb,
                  n_head = n_head,n_layer = n_layer, LoRA = False,FAVOR = FAVOR, random_features = random_features)

    learning_rate = LinearSchedule(warmup_steps = 100,decay = 8000,lr = 6.25e-5)
    
    optimizer= tf.keras.optimizers.Adam(
                                            learning_rate = learning_rate,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-08,
                                            weight_decay=0.01,
                                            clipvalue=1,
                                            global_clipnorm=1,
                                            jit_compile=False,
                                        )
    

    for i in val.take(1):x = i

    h = model(x)
    model.summary()
    
    c = 0
    names = {}
    
    model = load_weights(model,n_ctx = n_ctx,n_special = 0 , n_embd = n_embd, freeze_emb = freeze_emb ,weights_shapes_path =  data_dir, weights_path = data_dir, names_path = data_dir)
    start = time.time()
    for epoch in range(epochs):
        train = train_ds.shuffle(5000).batch(batch)
        
        l_cum = np.zeros(10)
  
        for step, x_batch_train in enumerate(train):

            with tf.GradientTape() as tape:

                H = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                l1 = lm_loss(x_batch_train[0], H[0], H[1],n_ctx = n_ctx)
                train_loss = tf.reduce_mean(l1)


            grads = tape.gradient(train_loss , model.trainable_weights)


            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            l_cum[0] = tf.reduce_mean(l1)
            l_cum = np.roll(l_cum, 1)

            sys.stdout.write('\r'+("epoch: " + str(epoch+1) + " step: " +str(step+1) + "/" + str(train.cardinality().numpy()) + " loss: " + str(np.sum(l_cum)/25) + " memory: " + str(round(tf.config.experimental.get_memory_info('GPU:0')["current"]*1e-6)) + "   "))
            
    end = time.time()
    print(end - start)
    print(round(tf.config.experimental.get_memory_info('GPU:0')["peak"]*1e-6) )

    for I,i in enumerate(model.layers):
        for J,j in enumerate(model.layers[I].weights):
            np.save("/home/borz/Desktop/proj/weights_FAVOR/"+str(c)+".npy", model.layers[I].weights[J].numpy(), allow_pickle=False)
            names[str(c)] = model.layers[I].weights[J].name
            # print(model.layers[I].weights[J].numpy())
            c = c+1
            
    import json

    with open('/home/borz/Desktop/proj/weights_FAVOR/data.json', 'w') as fp:
        json.dump(names, fp)
    
        
    



