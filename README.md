# GPT1 with causal FAVOR Attention 
This repository contains a Tensorflow 2 implementation of GPT1 with causal FAVOR (Fast Attention Via positive Orthogonal Random features).  
The model is compatible with the original weights from the paper:"Improving Language Understanding
by Generative Pre-Training". The weights can be found on: https://github.com/openai/finetune-transformer-lm. This repository was mainly made for benchmarking and testing purposes.

 <ol>
  <li><a href="#head1">What is FAVOR.</a></li>
  <li><a href="#head2">Differences from the original paper.</a></li>
  <li><a href="#head3">How to use.</a><ul>
  <li><a href="#head4">Base model.</a></li>
  <li><a href="#head5">Some useful tools.</a></li>


## <a name="head1"></a>What is FAVOR Attention
FAVOR or "Fast Attention Via positive Orthogonal Random features" is one way to make Attention scale linearly with sequence length. It works by doing the following approximation: <br />

$A = exp(Q\cdot K) \approx Q{}' \cdot K{}'$
<br /><br />
This allows one to calculate attention without using the softmax function. The advantage of doing so is that we don't need to save the A term in memory (which size is quadratic in sequence length).
This approximation is unconditional (while other methods mostly abuse some sort of sparsity conditions in the attention matrix) and does not fundamentally change how attention is calculated. As a result, it also is backwards compatible with already pretrained Transformer models, with a little bit of fine-tuning.
For more theoretical details, see the paper: https://arxiv.org/pdf/2009.14794.pdf

## <a name="head2"></a>Differences from the original paper.
I made some minor changes from the original paper due to performance reasons. The Prefix sums are calculated differently. The original implementation uses a sum of sums which doesn't work well in any machine learning framework, see at:https://github.com/google-research/google-research/issues/1406. Keep in mind that this Implementation is still lacking in regards to memory usage. There seems to be no way around directly coding it in c++ and CUDA if one desires an efficient implementation.
I used an initializer for the calculation of the random_features which is much faster than the original implementation, however you can not have more random features than emb_dim / n_heads (which for this use case doesn't matter). I also included an option for low rank training for testing purposes.

## <a name="head3"></a>How to use<br />
### <a name="head4"></a>Base model:
The base model can be imported from the Model folder:<br />

from Model import Base_GPT
<br /><br />
**Base_GPT( n_vocab,
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
        random_features = 32)**
        
#### inputs: 
A tuple of tokens and masks with shapes 
(<tf.Tensor: shape=(batch_size, n_ctx, 2), dtype=int> , <tf.Tensor: shape=(batch_size, n_ctx), dtype=float32>)<br />
Because the gpt1 embedding is learned during training the model needs positional tokens in addition to the tokenized text. A tokenizer can be found in the TXT folder.

#### outputs: 
A tuple of logits and mask with shapes
(<tf.Tensor: shape=((n_ctx-1)*batch_size, n_vocab+n_ctx), dtype=float32>,<tf.Tensor: shape=(batch_size, n_ctx), dtype=float32>).
The language model loss is calculated by taking the cross entropy between labels and logits. A loss funtion can be found in the Layers folder under unil.py.

#### Parameters: 
n_vocab : int<br />
Base vocabulary token size.<br />
n_special : int<br />
Number of special tokens.<br />
n_ctx : int<br />
Maximum sequence length.<br />
n_embd : int<br />
Embedding dimension.<br />
n_head : int <br />
Number of attention heads.<br />
pdrop,rdrop : float<br />
Attention head dropout and merged head dropout respectively.<br />
mdrop : float<br />
Dropout of the final linear layer.<br />
train : bool<br />
Enables/disables dropout.<br />
freeze_emb : bool<br />
Whether to freeze the embedding layer (with exception of the special tokens) during training.<br />
scale : bool<br />
Whether to normalize the attention matrix<br />
LoRA : bool<br />
Whether to use low rank adaptation for training (see https://www.microsoft.com/en-us/research/publication/lora-low-rank-adaptation-of-large-language-models/)<br />
lora_dim : int<br />
Low rank dimension<br />
FAVOR : bool<br />
Whether to use causal FAVOR Attention.<br />
random_features : int<br />
Number of random orthogonal features.<br />
## <a name="head5"></a>Some useful tools:

To load the weights from OpenAi to the model you can use:

**Layers.load_weights(model,n_ctx = 512,n_special = 0, n_embd = 768, freeze_emb = True,weights_shapes_path =  "./weights", weights_path = "./weights", names_path = "./weights",LoRA = False,FAVOR = False)**<br />
Used to load the pretrained model weights from https://github.com/openai/finetune-transformer-lm/tree/master/model into the model.<br /><br />
model : tensorflow model<br />
model instance to load weights into.<br />
weights_shapes_path : str<br />
Path of the folder containing params_shapes.json found on the open ai github.<br />
weights_path : str<br />
Path of the folder containing params_n.npy found on the open ai github.<br />
names_path : str<br />
Path of the folder containing names_emb_f.json and names_emb.json found on under the /Files folder.<br />
FAVOR : bool<br />
Whether to use causal FAVOR Attention. If set to true a different set of weight parameters will be loaded to the model. A script is provided to finetune the original weights to FAVOR weights and save them.<br /><br />

Additionally there are two scripts: train_finetune.py can be used to fine-tune the model to use FAVOR attention, around 50mb of text corpus is enough to yield decent results.

Train_roc.py does the same cloze Benchmark that was done on the OpenAi paper (https://www.cs.rochester.edu/nlp/rocstories/). You should get around 87% accuracy with the original model and around 85% accuracy with FAVOR if you did the fine-tuning. For the  fine-tuning I strongly recommend to use the same corpus the model was initially trained on.















