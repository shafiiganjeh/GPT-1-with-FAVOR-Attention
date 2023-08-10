import tensorflow as tf

import sys
sys.path.append("..")

from Layers import*

class GPT(tf.keras.Model):
    def __init__(
        self,
        n_vocab,
        n_special,
        n_ctx,
        n_embd,
        clf_token,
        lora_dim = 4,
        n_head = 12,
        n_layer = 12,
        pdrop = .1,
        rdrop = .1,
        mdrop = .1,
        clf_pdrop = .1,
        train = True,
        freeze_emb = False,
        scale = True,
        LoRA = False,
        FAVOR = False,
        random_features = None
    ):
        super().__init__()
        self.train = train  
        self.n_head = n_head 
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.clf_token = clf_token
        self.clf_pdrop = clf_pdrop
        
        self.outp = None
        
        self.embed = embedding(n_vocab = n_vocab ,n_special = n_special, 
                                n_ctx = self.n_ctx, n_embd = self.n_embd,freeze_emb = freeze_emb)
        
        
        self._block = [None]*self.n_layer
        
        for i in range(self.n_layer ):
            self._block[i] = block(train = self.train,n_head = self.n_head,
                          mdrop = mdrop,pdrop = pdrop,
                          rdrop = rdrop,scale = scale,
                          LoRA = LoRA,lora_dim = lora_dim,FAVOR = FAVOR,
                          random_features = random_features,name_ = "block_"+str(i))
            
        self.clf = tf.keras.layers.Dense(1,bias_initializer = 'zeros',kernel_initializer = tf.random_normal_initializer(stddev=0.02, seed=123))

    
    def call(self, x):
        X = x[0]
        M = x[1]
        assert X.shape[-1] == 2 and X.shape[-2] == self.n_ctx , "input shape missmatch"
        
        M = tf.reshape(M, [-1, self.n_ctx])
        X = tf.reshape(X, [-1, self.n_ctx, 2])
        
        h = self.embed(X)[0]
        W = self.embed(X)[1]
        
        for i in range(self.n_layer):
            h = self._block[i](h)
            
        lm_h = tf.reshape(h[:, :-1], [-1, self.n_embd])
        lm_logits = tf.matmul(lm_h, W, transpose_b=True)
        
        clf_h = tf.reshape(h, [-1, self.n_embd])
        pool_idx = tf.cast(tf.argmax(tf.cast(tf.equal(X[:, :, 0], self.clf_token), tf.float32), 1), tf.int32)
        clf_h = tf.gather(clf_h, tf.range(shape_list(X)[0], dtype=tf.int32)*self.n_ctx+pool_idx)

        clf_h = tf.reshape(clf_h, [-1, 2, self.n_embd])
        if self.train and self.clf_pdrop > 0:
            shape = shape_list(clf_h)
            shape[1] = 1
            clf_h = tf.nn.dropout(clf_h, self.clf_pdrop, shape)
        clf_h = tf.reshape(clf_h, [-1, self.n_embd])
        clf_logits = self.clf(clf_h)
        clf_logits = tf.reshape(clf_logits, [-1, 2])
            

        return [lm_logits,M,clf_logits]

class Base_GPT(tf.keras.Model):
    def __init__(
        self,
        n_vocab,
        n_special,
        n_ctx,
        n_embd,
        n_head = 12,
        n_layer = 12,
        pdrop = .1,
        rdrop = .1,
        mdrop = .1,
        train = True,
        freeze_emb = True,
        scale = True,
        LoRA = False,
        lora_dim = 4,
        FAVOR = False,
        random_features = None
    ):
        super().__init__()
        self.train = train  
        self.n_head = n_head 
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer

        
        self.outp = None
        
        self.embed = embedding(n_vocab = n_vocab ,n_special = n_special, 
                                n_ctx = self.n_ctx, n_embd = self.n_embd,freeze_emb = freeze_emb)
        
        
        self._block = [None]*self.n_layer
        
        for i in range(self.n_layer ):
            self._block[i] = block(train = self.train,n_head = self.n_head,
                          mdrop = mdrop,pdrop = pdrop,
                          rdrop = rdrop,scale = scale,
                          LoRA = LoRA,lora_dim = lora_dim,FAVOR = FAVOR,
                          random_features = random_features,name_ = "block_"+str(i))

    
    def call(self, x):
        X = x[0]
        M = x[1]
        # assert X.shape[0] == self.n_ctx , "input shape missmatch"
        
        
        h = self.embed(X)[0]
        W = self.embed(X)[1]
        
        for i in range(self.n_layer):
            h = self._block[i](h)
            
        lm_h = tf.reshape(h[:, :-1], [-1, self.n_embd])
        lm_logits = tf.matmul(lm_h, W, transpose_b=True)
        

        return [lm_logits,M]

class MedtGPT(tf.keras.Model):
    def __init__(
        self,
        n_vocab,
        n_special,
        n_ctx,
        n_embd,
        clf_token,
        clf_token_nr,
        lora_dim = 4,
        n_head = 12,
        n_layer = 12,
        pdrop = .1,
        rdrop = .1,
        mdrop = .1,
        clf_pdrop = .1,
        train = True,
        freeze_emb = False,
        scale = True,
        LoRA = False,
        FAVOR = False,
        random_features = None
    ):
        super().__init__()
        self.train = train  
        self.n_head = n_head 
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.clf_pdrop = clf_pdrop
        
        self.outp = None
        
        self.embed = embedding(n_vocab = n_vocab ,n_special = n_special, 
                                n_ctx = self.n_ctx, n_embd = self.n_embd,freeze_emb = freeze_emb)
        
        
        self._block = [None]*self.n_layer
        
        for i in range(self.n_layer ):
            self._block[i] = block(train = self.train,n_head = self.n_head,
                          mdrop = mdrop,pdrop = pdrop,
                          rdrop = rdrop,scale = scale,
                          LoRA = LoRA,lora_dim = lora_dim,FAVOR = FAVOR,
                          random_features = random_features,name_ = "block_"+str(i))
            
        self.clf = tf.keras.layers.Dense(clf_token_nr,bias_initializer = 'zeros',kernel_initializer = tf.random_normal_initializer(stddev=0.02, seed=123))
        
        self.clf = tf.keras.layers.Dense(149,bias_initializer = 'zeros',kernel_initializer = tf.random_normal_initializer(stddev=0.02, seed=123))
        
        self.lm_cls = clf_token
        
        
    def call(self, x):
        X = x[0]
        M = x[1]
        assert X.shape[-1] == 2 and X.shape[-2] == self.n_ctx , "input shape missmatch"
        
        M = tf.reshape(M, [-1, self.n_ctx])
        X = tf.reshape(X, [-1, self.n_ctx, 2])
        
        h = self.embed(X)[0]
        W = self.embed(X)[1]
        
        for i in range(self.n_layer):
            h = self._block[i](h)
            
        lm_h = tf.reshape(h[:, :-1], [-1, self.n_embd])
        lm_logits = tf.matmul(lm_h, W, transpose_b=True)

        X2 = tf.reshape(x[0][:,:,0], [1, -1])
        pool_idx = tf.where(tf.reduce_any(tf.math.equal(self.lm_cls,X2),0))
        pool_idx = tf.squeeze(pool_idx)
        
        clf_logits = tf.reshape(h,[-1,self.n_embd])
        clf_logits = tf.gather(clf_logits,pool_idx)
        shape = shape_list(x[0])[0]
        clf_logits = tf.reshape(clf_logits,[shape,-1])
        
        if self.train and self.clf_pdrop > 0:
            clf_logits = tf.nn.dropout(clf_logits, self.clf_pdrop)
            
        clf_logits = self.clf(clf_logits)
        clf_logits = tf.reshape(clf_logits, [-1, 149])
        clf_logits = tf.math.sigmoid(clf_logits,-1)

        return [lm_logits,M,clf_logits]
    
    
    
