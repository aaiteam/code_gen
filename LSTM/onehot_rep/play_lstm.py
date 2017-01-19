import pickle

import chainer.links as L
import chainer.serializers as S
import numpy as np
import six

import pretrain_lstm

print ("loading the data...")

with open('./python_corpus.pkl', 'rb') as f:
    data = pickle.load(f)

vocab = data['codebook'] + ['<soc>', '<eoc>']
vocab2index = {x: i for i, x in enumerate(vocab)}
index2vocab = {i: x for i, x in enumerate(vocab)}

n_vocab = len(vocab)

rnn = pretrain_lstm.ActionValue2(n_vocab, 650)
model = L.Classifier(rnn)
S.load_npz('./result/lstm_model2.npz', model)

print ("Finished loading!")

def make_comment(word):
    if word not in vocab:
        return '{} is not in vocab'.format(word)
    rnn.reset_state()
    init = rnn(np.asarray([vocab2index['<soc>']], dtype=np.int32), train=False)
    comment = word
    a = rnn(np.asarray([vocab2index[word]], dtype=np.int32), train=False)

    while True:
        now = index2vocab[int(np.argmax(a.data))]
        if now == '<eoc>':
            break
        elif len(comment) > 20:
            break
        comment += now
        a = rnn(np.asarray([vocab2index[now]], dtype=np.int32), train=False)

    return comment


try:
    while True:
        q = six.moves.input('>> ')
        print(make_comment(q))

except EOFError:
    pass
