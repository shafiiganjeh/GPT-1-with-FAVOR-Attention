import argparse
import sys
import tensorflow as tf
from TXT import*
from Model import GPT
from Layers import ExpSchedule,LinearSchedule,load_weights,lm_loss,cl_loss
from sklearn.metrics import accuracy_score

import subprocess as sp
import os

import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prep', type=bool, default=False)
    parser.add_argument('--lora', type=bool, default=False)
    parser.add_argument('--FAVOR', type=bool, default=False)
    parser.add_argument('--random_features', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='Files/')
    parser.add_argument('--data_dir', type=str, default='Files/')
    parser.add_argument('--val_name', type=str, default='cloze_test_val__spring2016 - cloze_test_ALL_val.csv')
    parser.add_argument('--test_name', type=str, default='cloze_test_test__spring2016 - cloze_test_ALL_test.csv')
    parser.add_argument('--n_ctx', type=int, default=77)
    parser.add_argument('--lora_dim', type=int, default=4  )
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--freeze_emb', type=bool, default=True)
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--n_train', type=int, default=1497)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--l', type=float, default=1)
    args = parser.parse_args()
    print(args)
    globals().update(args.__dict__)
    
    text_encoder = TextEncoder(data_dir + 'encoder_bpe_40000.json',data_dir + 'vocab_40000.bpe')
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    
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
    
    if prep:
        n_special = len(text_encoder.encoder) - n_vocab

        dt = rocstories(path_val = data_dir + val_name, path_test = data_dir + test_name, n_train = n_train, n_valid = n_valid)
        (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3) = encode_dataset(dt, encoder = text_encoder)
        
        max_len = n_ctx//2-2
        n_ctx = min(max([len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(trX1, trX2, trX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(vaX1, vaX2, vaX3)]+[len(x1[:max_len])+max(len(x2[:max_len]), len(x3[:max_len])) for x1, x2, x3 in zip(teX1, teX2, teX3)])+3, n_ctx)
        
        trX, trM = transform_roc(trX1, trX2, trX3,clf_token = clf_token,n_vocab = n_vocab,n_special = n_special,n_ctx = n_ctx ,encoder = encoder,max_len = max_len)
        vaX, vaM = transform_roc(vaX1, vaX2, vaX3,clf_token = clf_token,n_vocab = n_vocab,n_special = n_special,n_ctx = n_ctx ,encoder = encoder,max_len = max_len)
        
        validation = tf.data.Dataset.from_tensor_slices(((vaX,vaM), vaY))
        train_ds = tf.data.Dataset.from_tensor_slices(((trX,trM), trY))
        train_ds.save(save_dir + "train")
        validation.save(save_dir + "val")
        
        del dt,trX, trM,vaX, vaM,trX1, trX2, trX3, trY, vaX1, vaX2, vaX3, vaY, teX1, teX2, teX3
    else:
        n_ctx = 77
        n_special = 3
        train_ds = tf.data.Dataset.load(save_dir + "/train",element_spec=((tf.TensorSpec(shape=(2, 77, 2), dtype=tf.int32, name=None), tf.TensorSpec(shape=(2, 77), dtype=tf.float32, name=None)), tf.TensorSpec(shape=(), dtype=tf.int32, name=None)))
        validation = tf.data.Dataset.load(save_dir + "/val",element_spec=((tf.TensorSpec(shape=(2, 77, 2), dtype=tf.int32, name=None), tf.TensorSpec(shape=(2, 77), dtype=tf.float32, name=None)), tf.TensorSpec(shape=(), dtype=tf.int32, name=None)))
        
    validation = validation.batch(batch)

    model = GPT(n_vocab = n_vocab,n_special = n_special, n_ctx = n_ctx, 
                 n_embd = n_embd,clf_token = clf_token,train = True,freeze_emb = freeze_emb,
                 n_head = n_head,n_layer = n_layer, LoRA = lora,lora_dim = lora_dim,
                 FAVOR = FAVOR, random_features = random_features)
    if FAVOR:
        if lora:
            learning_rate = ExpSchedule(warmup_steps = 20,decay = .008,lr = 1e-3)
            weight_decay = 0.01
        else:
            learning_rate = LinearSchedule(warmup_steps = 4,decay = 93*8,lr = 3.5e-5)
            weight_decay = None
    else:
        if lora:
            learning_rate = ExpSchedule(warmup_steps = 20,decay = .005,lr = 6.25e-4)
            weight_decay = 0.01
        else:
            learning_rate = LinearSchedule(warmup_steps = 4,decay = 93*5,lr = 6.25e-5)
            weight_decay = None
        
    # element_spec=((TensorSpec(shape=(16, 2, 77, 2), dtype=tf.int32),TensorSpec(shape=(16, 2, 77), dtype=tf.float32)),TensorSpec(shape=(16,), dtype=tf.int32))
    # ((TensorSpec(shape=(16, 2, 77, 2), dtype=tf.int32),TensorSpec(shape=(16, 2, 77), dtype=tf.float32)),TensorSpec(shape=(16,), dtype=tf.int32))

    optimizer= tf.keras.optimizers.Adam(
                                            learning_rate = learning_rate,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-08,
                                            weight_decay=weight_decay,
                                            clipvalue=1,
                                            global_clipnorm=1,
                                            jit_compile=True,
                                        )
  

    for i in validation.take(1):x = i

    h = model(x[0])
    model.summary()

    model = load_weights(model,n_ctx = n_ctx,n_special = n_special , n_embd = n_embd,
                          freeze_emb = freeze_emb ,weights_shapes_path =  data_dir, 
                          weights_path = data_dir, names_path = data_dir,FAVOR = FAVOR, LoRA = lora)
    
    start = time.time()
    for epoch in range(epochs):
        train = train_ds.shuffle(5000).batch(batch)

        print(" val:"+acc_(model,validation))
        print(" train:"+acc_(model,train))
        
        l_cum = np.zeros(10)
  
        for step, (x_batch_train, y_batch_train) in enumerate(train):

            with tf.GradientTape() as tape:

                H = model(x_batch_train, training=True)  # Logits for this minibatch

                # Compute the loss value for this minibatch.
                l1 = lm_loss(x_batch_train[0], H[0], H[1],n_ctx = n_ctx)
                l2 = cl_loss(y_batch_train, H[2])
                train_loss = tf.reduce_mean(l2) + l* tf.reduce_mean(l1)


            grads = tape.gradient(train_loss , model.trainable_weights)


            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            l_cum[0] = tf.reduce_mean(l2)
            l_cum = np.roll(l_cum, 1)

            sys.stdout.write('\r'+("epoch: " + str(epoch+1) + " step: " +str(step+1) + "/" + str(train.cardinality().numpy()) + " loss: " + str(np.sum(l_cum)/25) + " memory: " + str(round(tf.config.experimental.get_memory_info('GPU:0')["current"]*1e-6)) + "   "))
            
    print(" val:"+acc_(model,validation))
    print(" train:"+acc_(model,train))
end = time.time()
print(end - start)
print(round(tf.config.experimental.get_memory_info('GPU:0')["peak"]*1e-6) )
        
    



