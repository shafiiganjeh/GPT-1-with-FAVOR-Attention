import os
import csv
import numpy as np
import glob
import re
from tqdm import tqdm
import tensorflow as tf

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

seed = 3535999445

def transform_roc(X1, X2, X3, n_ctx,encoder,max_len,clf_token,n_vocab,n_special):
    n_batch = len(X1)
    xmb = np.zeros((n_batch, 2, n_ctx, 2), dtype=np.int32)
    mmb = np.zeros((n_batch, 2, n_ctx), dtype=np.float32)
    start = encoder['_start_']
    delimiter = encoder['_delimiter_']
    for i, (x1, x2, x3), in enumerate(zip(X1, X2, X3)):
        x12 = [start]+x1[:max_len]+[delimiter]+x2[:max_len]+[clf_token]
        x13 = [start]+x1[:max_len]+[delimiter]+x3[:max_len]+[clf_token]
        l12 = len(x12)
        l13 = len(x13)
        xmb[i, 0, :l12, 0] = x12
        xmb[i, 1, :l13, 0] = x13
        mmb[i, 0, :l12] = 1
        mmb[i, 1, :l13] = 1
    xmb[:, :, :, 1] = np.arange(n_vocab+n_special, n_vocab+n_special+n_ctx)
    return xmb, mmb


def _rocstories(path):
    with open(path) as f:
        f = csv.reader(f)
        st = []
        ct1 = []
        ct2 = []
        y = []
        for i, line in enumerate(tqdm(list(f), ncols=80, leave=False)):
            if i > 0:
                s = ' '.join(line[1:5])
                c1 = line[5]
                c2 = line[6]
                st.append(s)
                ct1.append(c1)
                ct2.append(c2)
                y.append(int(line[-1])-1)
        return st, ct1, ct2, y
    

def rocstories(path_test,path_val, n_train=1497, n_valid=374):
    storys, comps1, comps2, ys = _rocstories(os.path.join(path_val))
    teX1, teX2, teX3, _ = _rocstories(os.path.join(path_test))
    tr_storys, va_storys, tr_comps1, va_comps1, tr_comps2, va_comps2, tr_ys, va_ys = train_test_split(storys, comps1, comps2, ys, test_size=n_valid, random_state=seed)
    trX1, trX2, trX3 = [], [], []
    trY = []
    for s, c1, c2, y in zip(tr_storys, tr_comps1, tr_comps2, tr_ys):
        trX1.append(s)
        trX2.append(c1)
        trX3.append(c2)
        trY.append(y)

    vaX1, vaX2, vaX3 = [], [], []
    vaY = []
    for s, c1, c2, y in zip(va_storys, va_comps1, va_comps2, va_ys):
        vaX1.append(s)
        vaX2.append(c1)
        vaX3.append(c2)
        vaY.append(y)
    trY = np.asarray(trY, dtype=np.int32)
    vaY = np.asarray(vaY, dtype=np.int32)
    return (trX1, trX2, trX3, trY), (vaX1, vaX2, vaX3, vaY), (teX1, teX2, teX3)

def create_pretrain_dataset(text_encoder = None,txt_files = "/Files", save_path = "/Files",length = 512,seperator = '\n',ignore_seperator = False ,skip = 1000):
    
    path = txt_files + "/*.txt"
    files = glob.glob(path)
    
    assert len(files) > 0, "No files"
    
    dataset = []
    k = 0
    deleted = 0
    
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)
    
    for i in tqdm(files, ncols=80, leave=True):
        if k > 200:
            break
        f = open(i, "r")
        a = f.read()
        a = a[skip:]
        loc = [m.start() for m in re.finditer(seperator, a)]
        a = [[sen for sen in [a[loc[j]:loc[j+1]]]][0] for j in range(len(loc)-1)]
        if ignore_seperator:
            a = [i.replace(seperator, ' ') for i in a]
        a = text_encoder.encode([a[i] for i in range(len(a))], verbose=False)
        data = []
        temp = []
        for i in a:
            if len(i) > length:
                deleted = deleted +1
            else:
                if len(temp + i) > length:
                    data.append(temp)
                    temp = []
                else:
                    temp = temp + i
        if len(temp) > 0:
            data.append(temp)
            
        pos = np.arange(n_vocab,n_vocab+length,1)
        M = []
    
        for i,j in enumerate(data):
            if 0 > length - len(j):
                M.append([1]*length)
                data[i] = np.array(data[i][0:length],dtype=np.int32)
            else:
                M.append(np.array([1]*int(len(j))+[0]*int(length - len(j)),dtype=np.float32))
                data[i] = np.array(data[i]+[0]*int(length - len(j)),dtype=np.int32)
            data[i] = np.stack([data[i],pos],axis=1)
    
        dataset.append(tf.data.Dataset.from_tensor_slices((data,M)))
        k = k+1
    
    print("deleted passages: "+str(deleted))
    
    full_ds = dataset[0]
    for i in range(1,len(dataset),1):
        full_ds = full_ds.concatenate(dataset[i])
    
    tf.data.experimental.save(full_ds, save_path)
    return

