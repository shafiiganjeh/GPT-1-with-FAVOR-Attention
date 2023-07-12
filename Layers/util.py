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

# lr = 6.25e-5
class LinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps,decay, lr = 6.25e-5):
    super().__init__()
    self.lr = lr
    self.warmup_steps = warmup_steps
    self.decay = decay

  def __call__(self, step):

    arg1 = self.lr*(((self.warmup_steps+self.decay)-step)/self.decay)
    arg2 = (step/self.warmup_steps) * self.lr

    return tf.math.minimum(arg1, arg2)


class ExpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, warmup_steps,decay = .005,lr = 6.25e-5):
    super().__init__()

    self.lr = lr
    self.warmup_steps = warmup_steps
    self.decay = decay

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = (step/self.warmup_steps) * self.lr
    e = tf.math.maximum(0,step - self.warmup_steps)
    arg2 = tf.math.exp(-e*self.decay)*self.lr

    return tf.math.minimum(arg1, arg2)



def shape_list(x):
    ps = tf.shape(x)
    ts = tf.shape(x)
    return [ts[i] if ps[i] is None else ps[i] for i in range(len(ps))]


def load_weights(model,n_ctx = 77,n_special = 3, n_embd = 768, freeze_emb = True,weights_shapes_path =  "./weights", weights_path = "./weights", names_path = "./weights",LoRA = False,FAVOR = False):
    L = [[i.name for i in j.weights[:]] for j in model.layers[:]]
    # names = {}
    np.random.seed(123)
    shapes = json.load(open(weights_shapes_path+"/params_shapes.json"))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    if FAVOR: 
        init_params = [np.load(weights_path+"/{}.npy".format(n)) for n in range(147)]
        init_params = init_params[1:]
        for i in range(len(init_params)):
            init_params[i] = init_params[i].flatten()
    else:
        init_params = [np.load(weights_path+"/params_{}.npy".format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    init_params[0] = init_params[0][:n_ctx]
    if freeze_emb:
        init_params.insert(0, (np.random.randn(n_special, n_embd)*0.02).astype(np.float32))
        with open(names_path+'names_emb_f.json', 'r') as openfile:
            NAMES = json.load(openfile)
    else:
        init_params[0] = np.concatenate([init_params[1], (np.random.randn(n_special, n_embd)*0.02).astype(np.float32), init_params[0]], 0)
        del init_params[1]
        with open(names_path+'names_emb.json', 'r') as openfile:
            NAMES = json.load(openfile)
    assg = 0
    for n in range(len(init_params)):
        if LoRA == False:
           # print(NAMES[str(n)])
           T = NAMES[str(n)].replace("__lo_ra","") 
           # print(T)
        else:
           T = NAMES[str(n)] 
        for I,i in enumerate(L):
            for J,j in enumerate(i):
                if T in j:
                    model.layers[I].weights[J].assign(init_params[n])
                    assg = assg + 1
    print("weights assigned: " + str(assg) + "/" + str(len(init_params)))           

    return model

def save_weights_lora(model, path):
    names = {}
    c = 0
    for I,i in enumerate(model.layers):
        for J,j in enumerate(model.layers[I].weights):
            if ("LoRA" in model.layers[I].weights[J].name):
                np.save(path+"/lora"+str(c)+".npy", model.layers[I].weights[J].numpy(), allow_pickle=False)
                names[str(c)] = model.layers[I].weights[J].name[model.layers[I].weights[J].name.find("/"):]
                c = c+1
            elif ("embedding" in model.layers[I].weights[J].name) and model.layers[I].weights[J].trainable:
                np.save(path+"/lora"+str(c)+".npy", model.layers[I].weights[J].numpy(), allow_pickle=False)
                names[str(c)] = model.layers[I].weights[J].name[model.layers[I].weights[J].name.find("/"):]
                c = c+1
    with open(path + '/names.json', 'w') as fp:
        json.dump(names, fp)
    return names


def load_weights_lora(model, path):
    L = [[i.name for i in j.weights[:]] for j in model.layers[:]]
    with open(path+'/names.json', 'r') as openfile:
        NAMES = json.load(openfile)
    assg = 0
    init_params = [np.load(path+"/lora{}.npy".format(n)) for n in range(len(NAMES))]
        
    for n in range(len(NAMES)):
        T = NAMES[str(n)]
        for I,i in enumerate(L):
            for J,j in enumerate(i):
                if T in j:
                    model.layers[I].weights[J].assign(init_params[n])
                    assg = assg + 1
    print("weights assigned: " + str(assg) + "/" + str(len(init_params)))  
    
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




