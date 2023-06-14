import tensorflow as tf
import numpy as np
import json


def gelu(x):
    return tf.keras.activations.gelu(x)


def swish(x):
    return tf.keras.activations.swish(x)

act_fns = {
    'relu':tf.nn.relu,
    'swish':swish,
    'gelu':gelu
}


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps,decay, lr = 6.25e-5):
    super().__init__()
    self.lr = lr
    self.warmup_steps = warmup_steps
    self.decay = decay

  def __call__(self, step):

    arg1 = self.lr*(((self.warmup_steps+self.decay)-step)/self.decay)
    arg2 = (step/self.warmup_steps) * self.lr

    return tf.math.minimum(arg1, arg2)


def shape_list(x):
    ps = tf.shape(x)
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def load_weights(model,n_ctx = 77,n_special = 3, n_embd = 768, freeze_emb = True,weights_shapes_path =  "./weights", weights_path = "./weights"):
    
    np.random.seed(123)
    shapes = json.load(open(weights_shapes_path+"/params_shapes.json"))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(weights_path+"/params_{}.npy".format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    if freeze_emb:
        init_params.insert(0, (np.random.randn(n_special, n_embd)*0.02).astype(np.float32))
    else:
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
        del init_params[1]
    c = 0
    for i in range(len(model.layers)):
        if c == len(init_params):
            break
        for j in range(len(model.layers[i].weights)):
            model.layers[i].weights[j].assign(init_params[c])
            c = c+1
    
    return model

@tf.function
def lm_loss(label, pred, M,n_ctx = 512):
    label = tf.reshape(label, [-1, n_ctx, 2])
    lm_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.reshape(label[:, 1:, 0], [-1]))
    lm_losses = tf.reshape(lm_losses, [shape_list(label)[0], shape_list(label)[1]-1])
    lm_losses = tf.reduce_sum(lm_losses*M[:, 1:], 1)/tf.reduce_sum(M[:, 1:], 1)
    return lm_losses

@tf.function
def cl_loss(label, pred):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)



