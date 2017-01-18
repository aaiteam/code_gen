# coding: utf-8

# http://qiita.com/chachay/items/052406176c55dd5b9a6a
#
# http://www.monthly-hack.com/entry/2016/10/12/121613

# Please download data below
## https://drive.google.com/drive/u/2/folders/0B33vEXpXOGfmUmFFUVhsY3R2elU


import argparse
import pickle

import chainer
import chainer.links as L
import numpy as np
from chainer import training, Chain, Variable
from chainer.training import extensions


class ActionValue(Chain):
    def __init__(self, n_vocab, n_units):
        super(ActionValue, self).__init__(
                embed=L.EmbedID(n_vocab, n_units),
                lstm=L.LSTM(in_size=n_units, out_size=n_units),
                # fc=L.Linear(in_size=10, out_size=10),
                q_value=L.Linear(n_units, n_vocab)
        )
        self.h_sum = None
        self.seq_len = np.asarray([0])

    def __call__(self, x, train=True):
        h = self.embed(x)
        h = self.lstm(h)
        h = self.q_value(h)
        return h

    def q_function(self, action):
        x = self.embed(Variable(np.asarray([np.int32(action + 1)])))
        lstm_res = self.lstm(x)
        res = self.q_value(lstm_res)
        # fc_res = F.relu(self.fc(lstm2_res))
        return res

    def reset_state(self):
        self.lstm.reset_state()
        # self.lstm1.reset_state()
        # self.lstm2.reset_state()
        self.h_sum = None
        self.seq_len = np.asarray([0])

    def set_state(self, actions):
        self.reset_state()
        for action in actions:
            self.q_function(action)


class ParallelSequentialIterator(chainer.dataset.Iterator):
    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epoch = 0
        self.is_new_epoch = False
        self.repeat = repeat
        length = len(dataset)
        self.offsets = [i * length // batch_size for i in range(batch_size)]
        self.iteration = 0

    def __next__(self):
        length = len(self.dataset)
        if not self.repeat and self.iteration * self.batch_size >= length:
            raise StopIteration
        cur_words = self.get_words()
        self.iteration += 1
        next_words = self.get_words()

        epoch = self.iteration * self.batch_size // length
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch

        return list(zip(cur_words, next_words))

    @property
    def epoch_detail(self):
        return self.iteration * self.batch_size / len(self.dataset)

    def get_words(self):
        return [self.dataset[(offset + self.iteration) % len(self.dataset)]
                for offset in self.offsets]

    def serialize(self, serializer):
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)


class BPTTUpdater(training.StandardUpdater):
    def __init__(self, train_iter, optimizer, bprop_len, device):
        super(BPTTUpdater, self).__init__(
                train_iter, optimizer, device=device)
        self.bprop_len = bprop_len

    def update_core(self):
        loss = 0
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        # Progress the dataset iterator for bprop_len words at each iteration.
        for i in range(self.bprop_len):
            batch = train_iter.__next__()
            x = np.asarray([example[0] for example in batch], dtype=np.int32)
            t = np.asarray([example[1] for example in batch], dtype=np.int32)
            if self.device >= 0:
                x = chainer.cuda.to_gpu(x, device=self.device)
                t = chainer.cuda.to_gpu(t, device=self.device)
            loss += optimizer.target(chainer.Variable(x), chainer.Variable(t))

        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters


# Routine to rewrite the result dictionary of LogReport to add perplexity
# values
def compute_perplexity(result):
    result['perplexity'] = np.exp(result['main/loss'])
    if 'validation/main/loss' in result:
        result['val_perplexity'] = np.exp(result['validation/main/loss'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of examples in each mini-batch')
    parser.add_argument('--bproplen', '-l', type=int, default=10,
                        help='Number of words in each mini-batch '
                             '(= length of truncated BPTT)')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gradclip', '-c', type=float, default=5,
                        help='Gradient norm threshold to clip')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    parser.set_defaults(test=False)
    parser.add_argument('--unit', '-u', type=int, default=650,
                        help='Number of LSTM units in each layer')
    args = parser.parse_args()

    with open('./python_corpus.pkl', 'rb') as f:
        data = pickle.load(f)

    codebook = data['codebook'] + ['<soc>', '<eoc>']

    codebook2idx = {x: i for i, x in enumerate(codebook)}

    train_data = []
    for one_data in data['data_index']:
        train_data.append(codebook2idx['<soc>'])
        train_data += one_data
        train_data.append(codebook2idx['<eoc>'])

    train_data = np.array(train_data, dtype=np.int32)

    n_vocab = max(train_data) + 1  # train is just an array of integers
    print('#vocab =', n_vocab)
    n_train = len(train_data)  # train is just an array of integers
    print('#train =', n_train)
    train_iter = ParallelSequentialIterator(train_data, args.batchsize)

    # Prepare an RNNLM model
    # rnn = RNNForLM(n_vocab, args.unit)
    rnn = ActionValue(n_vocab, args.unit)
    model = L.Classifier(rnn)
    model.compute_accuracy = True  # we only want the perplexity
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # make the GPU current
        model.to_gpu()

    # Set up an optimizer
    optimizer = chainer.optimizers.SGD(lr=1.0)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradclip))

    # Set up a trainer
    updater = BPTTUpdater(train_iter, optimizer, args.bproplen, args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    interval = 10 if args.test else 500
    trainer.extend(extensions.LogReport(postprocess=compute_perplexity,
                                        trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(
            ['epoch', 'iteration', 'perplexity', 'main/accuracy']
    ), trigger=(interval, 'iteration'))
    # trainer.extend(extensions.ProgressBar(
    #         update_interval=1 if args.test else 1000))
    trainer.extend(extensions.snapshot())
    trainer.extend(extensions.snapshot_object(
            model, 'model_iter_{.updater.iteration}'))
    if args.resume:
        chainer.serializers.load_npz(args.resume, model)

    @training.make_extension(trigger=(500, 'iteration'))
    def save_model(trainer):
        chainer.serializers.save_npz('{}/{}'.format(
                args.out, 'lstm_model.npz'), model)

    trainer.extend(save_model)

    trainer.run()


if __name__ == '__main__':
    main()
