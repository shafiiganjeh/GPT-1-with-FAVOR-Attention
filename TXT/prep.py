import os
import csv
import numpy as np

from tqdm import tqdm

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

