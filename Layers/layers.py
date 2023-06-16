import tensorflow as tf
import numpy as np
from .util import shape_list,gelu,swish,act_fns

class norm(tf.keras.layers.Layer):
  def __init__(self, 
               axis = [-1],
               e = 1e-5
               ):
    super(norm, self).__init__()
    self.axis = axis
    self.e = e

  def build(self, x):
      n_state = x[-1]
      self.g = self.add_weight("g_n", shape = [n_state], initializer=tf.constant_initializer(1))
      self.b = self.add_weight("b_n", shape = [n_state], initializer=tf.constant_initializer(0))

  def call(self, x):
      u = tf.math.reduce_mean(x, axis = self.axis, keepdims=True)
      s = tf.math.reduce_mean(tf.square(x-u), axis = self.axis, keepdims=True)
      x = (x - u) * tf.math.rsqrt(s + self.e)
      return x*self.g + self.b


class mask(tf.keras.layers.Layer):
  def __init__(self):
    super(mask, self).__init__()
    self.b = None

  def build(self, input_shape):
      n = input_shape[-1]
      self.b = tf.linalg.band_part(tf.ones([n, n]), -1, 0)
      self.b = tf.reshape(self.b, [1, 1, n, n])

  def call(self, inputs):
    return inputs*self.b + -1e9*(1-self.b)


class embedding(tf.keras.layers.Layer):
    def __init__(
        self,
        n_vocab,
        n_special,
        n_ctx,
        n_embd,
        pdrop = .1,
        freeze_emb = False,
        train = True,
    ):
        super(embedding, self).__init__()
        self.train = train    
        self.n_vocab = n_vocab
        self.n_special = n_special
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.pdrop = pdrop
        self.freeze_emb = freeze_emb


    def build(self, x):
        if self.freeze_emb:
            
            self.w_embed_f2 = self.add_weight("we_1",shape = [self.n_ctx, self.n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.02, seed=None),
                                     trainable = False)


            self.w_embed_t = self.add_weight("we_2",shape = [self.n_special, self.n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.02, seed=None),
                                     trainable = True)
            
            
            self.w_embed_f1 = self.add_weight("we_3",shape = [self.n_vocab, self.n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.02, seed=None),
                                     trainable = False)
            
        else :
            self.w_embed = self.add_weight("we", 
                                     shape = [self.n_vocab+self.n_special+self.n_ctx, self.n_embd], 
                                     initializer = tf.random_normal_initializer(stddev=0.02, seed=None),
                                     trainable = self.train)
        
    def embed(self, X, we):
        e = tf.gather(we, X)
        h = tf.reduce_sum(e, 2)
        return h
    
    def dropout(self, x):
        if self.train and self.pdrop > 0:
            x = tf.nn.dropout(x, self.pdrop)
        return x
        
    def call(self, X):
        if self.freeze_emb:
            emb = tf.keras.layers.Concatenate(axis = 0)([self.w_embed_f1, self.w_embed_t,self.w_embed_f2])
            h = self.embed(X, emb)
            return [h,emb]
        else:
            emb = self.dropout(self.w_embed)
            h = self.embed(X, emb)
            return [h,self.w_embed]
    
    
    
class conv1d(tf.keras.layers.Layer):
    def __init__(
        self,
        nf,
        rf = 1,
        w_init = tf.random_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(0),
        pad = 'VALID',
        train = False,
    ):
        super(conv1d, self).__init__()
        self.rf = rf
        self.nf = nf
        self.w_init = w_init
        self.b_init = b_init
        self.pad = pad
        self.train = train

    def build(self, x):
        self.nx = x[-1]
        self.w = self.add_weight("w", 
                                 shape = [self.rf, self.nx, self.nf], 
                                 initializer = self.w_init,trainable = self.train)
        self.b = self.add_weight("b", 
                                 shape = [self.nf], 
                                 initializer = self.b_init,trainable = self.train)
        
    def call(self, x):
        
        
        if self.rf == 1: 

            c = tf.einsum('ble,reo->blo', x, self.w)+self.b
            
        else: 
            c = tf.nn.conv1d(x, self.w, stride=1, padding = self.pad)+self.b
        return c
    
    
    
class MHA(tf.keras.layers.Layer):
    def __init__(
        self,
        num_heads,
        key_dim,
        pdrop = 0.0,
        rdrop = 0.0,
        scale = False,
        train = False,
        LoRA = True,
        **kwargs,
    ):
        super(MHA,self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.pdrop = pdrop
        self.rdrop = rdrop
        self.scale = scale
        self.train = train
        self.LoRA = LoRA
        
    def build(self, x):
        assert self.key_dim % self.num_heads == 0, "K/Q dimension not divisible by number of heads"
        if self.LoRA:
            self.conv_inp = conv1d_LoRA(nf = self.key_dim*3, train = self.train)
            self.conv_out = conv1d_LoRA(nf = self.key_dim, train = self.train)
        else:
            self.conv_inp = conv1d(nf = self.key_dim*3, train = self.train)
            self.conv_out = conv1d(nf = self.key_dim, train = self.train)
            
        self.mask = mask()
        
    def split_states(self,x, n):
        x_shape = shape_list(x)
        m = x_shape[-1]
        new_x_shape = x_shape[:-1]+[n, m//n]
        return tf.reshape(x, new_x_shape)
    
    def dropout(self, x,drop):
        if self.train and drop > 0:
            x = tf.nn.dropout(x, drop)
        return x
    
    def split_heads(self,x, n, k=False):
        return tf.transpose(self.split_states(x, n), [0, 2, 1, 3])
        
    def _attn(self,q, k, v, train=False):
        w = tf.einsum('...ij,...kj->...ik', q, k)
        if self.scale:
            n_state = shape_list(v)[-1]
            w = w*tf.math.rsqrt(tf.cast(n_state, tf.float32))
        
        w = self.mask(w)
        w = tf.nn.softmax(w)
        
        w = self.dropout(w,drop = self.pdrop )
        
        a = tf.einsum('...ij,...jk->...ik', w, v)
        return a
    
    def merge_heads(self,x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2]+[np.prod(x_shape[-2:])]
        return tf.reshape(x, new_x_shape)
        
    def call(self, x):
        x = self.conv_inp(x)
        q, k, v = tf.split(x, 3, 2)
        
        q = self.split_heads(q, self.num_heads)
        k = self.split_heads(k, self.num_heads)
        v = self.split_heads(v, self.num_heads)
        
        a = self._attn(q, k, v, train = self.train)
        a = self.merge_heads(a)
        a = self.conv_out(a)
        a = self.dropout(a,drop = self.rdrop )
        return a
    
    
    
class MLP(tf.keras.layers.Layer):
    def __init__(
        self,
        n_state,
        train,
        mdrop = 0.,
        LoRA = True,
        afn = 'gelu'
    ):
        super(MLP, self).__init__()
        self.afn = afn
        self.train = train
        self.n_state = n_state
        self.drop = mdrop
        self.LoRA = LoRA
        

    def build(self, x):
        self.nx = x[-1]
        self.act = act_fns[self.afn]
        if self.LoRA:
            self.c_fc = conv1d_LoRA(nf = self.n_state, train = self.train, rf = 1)
            self.c_proj = conv1d_LoRA(nf = self.nx, train = self.train, rf = 1)
        else:
            self.c_fc = conv1d(nf = self.n_state, train = self.train, rf = 1)
            self.c_proj = conv1d(nf = self.nx, train = self.train, rf = 1)

        
    def dropout(self, x, pdrop, train):
        if self.train and pdrop > 0:
            x = tf.nn.dropout(x, pdrop)
        return x
        
    def call(self, X):
        h = self.act(self.c_fc(X))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, self.drop, self.train)
        return h2
    
    
class block(tf.keras.layers.Layer):
  def __init__(self, 
               train,
               n_head,
               pdrop = .0,
               rdrop = .0,
               mdrop = .0,
               scale = True
               ):
    super(block, self).__init__()
    self.train = train
    self.pdrop = pdrop
    self.rdrop = rdrop
    self.mdrop = mdrop
    self.scale = scale
    self.n_head = n_head

  def build(self, x):
      nx = x[-1]
      self._mha = MHA(pdrop = self.pdrop, rdrop = self.rdrop,
                      key_dim = nx, num_heads = self.n_head, 
                      train = self.train, scale = self.scale)
      
      self.norm1 = norm()
      
      self._mlp = MLP(n_state = nx*4, train = self.train,mdrop = self.mdrop)
      
      self.norm2 = norm()

  def call(self, x):
      a = self._mha(x)
      n = self.norm1(a+x)
      m = self._mlp(n)
      h = self.norm2(m+n)
      return h  


class conv1d_LoRA(tf.keras.layers.Layer):
    def __init__(
        self,
        nf,
        rf = 1,
        R = 4,
        scale = 32,
        w_init = tf.random_normal_initializer(stddev=0.02),
        b_init = tf.constant_initializer(0),
        pad = 'VALID',
        train = False,
    ):
        super(conv1d_LoRA, self).__init__()
        self.rf = rf
        self.nf = nf
        self.w_init = w_init
        self.b_init = b_init
        self.pad = pad
        self.train = train
        self.R = R
        self.scale = scale 

    def build(self, x):
        self.nx = x[-1]
        self.w = self.add_weight("conv_w", 
                                 shape = [self.rf, self.nx, self.nf], 
                                 initializer = self.w_init,trainable = False)
        self.b = self.add_weight("conv_b", 
                                 shape = [self.nf], 
                                 initializer = self.b_init,trainable = self.train)
        
        self.A = self.add_weight("LoRA_A", 
                                 shape = [self.rf, self.R, self.nf], 
                                 initializer = self.b_init,trainable = self.train)
        
        self.B = self.add_weight("LoRA_B", 
                                 shape = [self.rf, self.nx, self.R], 
                                 initializer = tf.constant_initializer(0),trainable = self.train)
        
    def call(self, x):
        
        
        if self.rf == 1: 
            
            c = tf.einsum('ble,reo->blo', x, self.w + 
                          (self.scale/self.R)*tf.einsum('...ij,...jk->...ik',self.B,self.A)) + self.b
              
        else: 
            
            c = tf.nn.conv1d(x, self.w + 
                             (self.scale/self.R)*tf.einsum('...ij,...jk->...ik',self.B,self.A), stride=1, padding = self.pad) + self.b
            
        return c
    
    
    
    
    
    