import numpy as np
import mxnet as mx
from pprint import pprint
import lstm
import bucket_io


def get_vocab(file_name):
    with open(file_name, 'r+') as f:
        all_chars = list(f.read())
    print(all_chars)

    idx = 0
    the_vocab = {}
    for char in all_chars:
        if len(char) == 0:
            continue
        if not (char in the_vocab):
            the_vocab[char] = idx
            idx += 1
    print(the_vocab)
    print('all vocab is:', len(the_vocab))
    # for k, v in the_vocab.items():
    #     if v == 1 or v == 2:
    #         print(k)
    return the_vocab


def build_lstm():
    seq_len = 129
    # embedding dimension, which maps a character to a 256-dimension vector
    num_embed = 256
    # number of lstm layers
    num_lstm_layer = 3
    # hidden unit in LSTM cell
    num_hidden = 512

    symbol = lstm.lstm_unroll(
        num_lstm_layer,
        seq_len,
        len(vocab) + 1,
        num_hidden=num_hidden,
        num_embed=num_embed,
        num_label=len(vocab) + 1,
        dropout=0.2)


if __name__ == '__main__':
    vocab = get_vocab('obama.txt')
