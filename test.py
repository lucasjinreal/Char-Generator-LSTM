import os
import mxnet as mx
import lstm
import bucket_io
import numpy as np
import logging
from rnn_model import LSTMInferenceModel
import rnn_model
import bisect
logging.getLogger().setLevel(logging.DEBUG)

with open('obama.txt', 'r+') as f:
    print(f.read()[0: 1000])


def read_content(path):
    with open(path) as ins:
        return ins.read()


def build_vocab(path):
    content = list(read_content(path))
    idx = 1
    the_vocab = {}
    for word in content:
        if len(word) == 0:
            continue
        if not word in the_vocab:
            the_vocab[word] = idx
            idx += 1
    return the_vocab


def text2id(sentence, the_vocab):
    words = list(sentence)
    return [the_vocab[w] for w in words if len(w) > 0]

vocab = build_vocab('obama.txt')
print('vocab size = ', len(vocab))


seq_len = 129
num_embed = 256
num_lstm_layer = 3
num_hidden = 512

symbol = lstm.lstm_unroll(
    num_lstm_layer,
    seq_len,
    len(vocab) + 1,
    num_hidden=num_hidden,
    num_embed=num_embed,
    num_label=len(vocab) + 1,
    dropout=0.2
)

batch_size = 32

init_c = [('l%d_init_c'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_h = [('l%d_init_h'%l, (batch_size, num_hidden)) for l in range(num_lstm_layer)]
init_states = init_c + init_h

data_train = bucket_io.BucketSentenceIter(
    "./obama.txt",
    vocab,
    [seq_len],
    batch_size,
    init_states,
    seperate_char='\n',
    text2id=text2id,
    read_content=read_content)

num_epoch = 1
# learning rate
learning_rate = 0.01


# Evaluation metric
def Perplexity(label, pred):
    loss = 0.
    for i in range(pred.shape[0]):
        loss += -np.log(max(1e-10, pred[i][int(label[i])]))
    return np.exp(loss / label.size)


def train_txt():
    model = mx.model.FeedForward(
        ctx=mx.gpu(0),
        symbol=symbol,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
        momentum=0,
        wd=0.0001,
        initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    model.fit(X=data_train,
              eval_metric=mx.metric.np(Perplexity),
              batch_end_callback=mx.callback.Speedometer(batch_size, 20),
              epoch_end_callback=mx.callback.do_checkpoint("obama"))


def MakeRevertVocab(vocab):
    dic = {}
    for k, v in vocab.items():
        dic[v] = k
    return dic


# make input from char
def MakeInput(char, vocab, arr):
    idx = vocab[char]
    tmp = np.zeros((1,))
    tmp[0] = idx
    arr[:] = tmp


# helper function for random sample
def _cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def _choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = _cdf(weights)
    x = np.random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


# we can use random output or fixed output by choosing largest probability
def MakeOutput(prob, vocab, sample=False, temperature=1.):
    if not sample:
        idx = np.argmax(prob, axis=1)[0]
    else:
        fix_dict = [""] + [vocab[i] for i in range(1, len(vocab) + 1)]
        scale_prob = np.clip(prob, 1e-6, 1 - 1e-6)
        rescale = np.exp(np.log(scale_prob) / temperature)
        rescale[:] /= rescale.sum()
        return _choice(fix_dict, rescale[0, :])
    try:
        char = vocab[idx]
    except:
        char = ''
    return char


def generate_txt():
    _, arg_params, __ = mx.model.load_checkpoint("obama", 75)

    # build an inference model
    model = rnn_model.LSTMInferenceModel(
        num_lstm_layer,
        len(vocab) + 1,
        num_hidden=num_hidden,
        num_embed=num_embed,
        num_label=len(vocab) + 1,
        arg_params=arg_params,
        ctx=mx.gpu(),
        dropout=0.2)
    seq_length = 600
    input_ndarray = mx.nd.zeros((1,))
    revert_vocab = MakeRevertVocab(vocab)
    # Feel free to change the starter sentence
    output = 'The United States'
    random_sample = False
    new_sentence = True

    ignore_length = len(output)

    for i in range(seq_length):
        if i <= ignore_length - 1:
            MakeInput(output[i], vocab, input_ndarray)
        else:
            MakeInput(output[-1], vocab, input_ndarray)
        prob = model.forward(input_ndarray, new_sentence)
        new_sentence = False
        next_char = MakeOutput(prob, revert_vocab, random_sample)
        if next_char == '':
            new_sentence = True
        if i >= ignore_length - 1:
            output += next_char
    print(output)

if __name__ == '__main__':
    generate_txt()