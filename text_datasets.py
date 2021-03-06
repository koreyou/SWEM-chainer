import csv
from contextlib import contextmanager
import glob
import io
import os
import shutil
import sys
import tarfile
import tempfile
from zipfile import ZipFile

import numpy
import chainer
import pandas as pd

from nlp_utils import make_vocab
from nlp_utils import normalize_text
from nlp_utils import transform_to_array
from nlp_utils import get_tokenizer

URL_DBPEDIA = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'  # NOQA
URL_IMDB = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
URL_OTHER_BASE = 'https://raw.githubusercontent.com/harvardnlp/sent-conv-torch/master/data/'  # NOQA


def download_dbpedia():
    path = chainer.dataset.cached_download(URL_DBPEDIA)
    tf = tarfile.open(path, 'r')
    # tf = (d.decode('utf-8') for d in tf)
    return tf


def read_dbpedia(tokenize, tf, split, shrink=1):
    dataset = []
    f = tf.extractfile('dbpedia_csv/{}.csv'.format(split))
    if sys.version_info > (3, 0):
        f = io.TextIOWrapper(f, encoding='utf-8')
    for i, (label, title, text) in enumerate(csv.reader(f)):
        if i % shrink != 0:
            continue
        label = int(label) - 1  # Index begins from 1
        tokens = tokenize(normalize_text(text))
        dataset.append((tokens, label))
    return dataset


def get_dbpedia(shrink=1, word_emb=None, stanfordcorenlp=None):
    tf = download_dbpedia()

    print('read dbpedia')
    with get_tokenizer(stanfordcorenlp) as tokenize:
        train = read_dbpedia(tokenize, tf, 'train', shrink=shrink)
        test = read_dbpedia(tokenize, tf, 'test', shrink=shrink)

    print('constract vocabulary based on frequency')
    vocab = make_vocab(train + test, max_vocab_size=500000)

    if word_emb is not None:
        print('load word embedding')
        emb, vocab = load_glove_vocab(word_emb, vocab=vocab.keys())
    else:
        emb = None

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab, emb


def download_imdb():
    path = chainer.dataset.cached_download(URL_IMDB)
    tf = tarfile.open(path, 'r')
    # To read many files fast, tarfile is untared
    path = tempfile.mkdtemp()
    tf.extractall(path)
    return path


def read_imdb(tokenize, path, split, shrink=1, fine_grained=False):
    fg_label_dict = {'1': 0, '2': 0, '3': 1, '4': 1,
                     '7': 2, '8': 2, '9': 3, '10': 3}

    def read_and_label(posneg, label):
        dataset = []
        target = os.path.join(path, 'aclImdb', split, posneg, '*')
        for i, f_path in enumerate(glob.glob(target)):
            if i % shrink != 0:
                continue
            with io.open(f_path, encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
            tokens = tokenize(normalize_text(text))
            if fine_grained:
                # extract from f_path. e.g. /pos/200_8.txt -> 8
                label = fg_label_dict[f_path.split('_')[-1][:-4]]
                dataset.append((tokens, label))
            else:
                dataset.append((tokens, label))
        return dataset

    pos_dataset = read_and_label('pos', 0)
    neg_dataset = read_and_label('neg', 1)
    return pos_dataset + neg_dataset


def get_imdb(shrink=1, word_emb=None, stanfordcorenlp=None,
             fine_grained=False):
    tmp_path = download_imdb()

    print('read imdb')
    with get_tokenizer(stanfordcorenlp) as tokenize:
        train = read_imdb(tokenize, tmp_path, 'train',
                          shrink=shrink, fine_grained=fine_grained)
        test = read_imdb(tokenize, tmp_path, 'test',
                         shrink=shrink, fine_grained=fine_grained)

    shutil.rmtree(tmp_path)

    print('constract vocabulary based on frequency')
    vocab = make_vocab(train)

    if word_emb is not None:
        print('load word embedding')
        emb, vocab = load_glove_vocab(word_emb, vocab=vocab.keys())
    else:
        emb = None

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab, emb


def download_other_dataset(name):
    if name in ['custrev', 'mpqa', 'rt-polarity', 'subj']:
        files = [name + '.all']
    elif name == 'TREC':
        files = [name + suff for suff in ['.train.all', '.test.all']]
    else:
        files = [name + suff for suff in ['.train', '.test']]
    file_paths = []
    for f_name in files:
        url = os.path.join(URL_OTHER_BASE, f_name)
        path = chainer.dataset.cached_download(url)
        file_paths.append(path)
    return file_paths


def read_other_dataset(tokenize, path, shrink=1):
    dataset = []
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for i, l in enumerate(f):
            if i % shrink != 0 or not len(l.strip()) >= 3:
                continue
            label, text = l.strip().split(None, 1)
            label = int(label)
            tokens = tokenize(normalize_text(text))
            dataset.append((tokens, label))
    return dataset


def get_other_text_dataset(name, shrink=1, word_emb=None,
                           stanfordcorenlp=None, seed=777):
    assert(name in ['TREC', 'stsa.binary', 'stsa.fine',
                    'custrev', 'mpqa', 'rt-polarity', 'subj'])
    datasets = download_other_dataset(name)
    with get_tokenizer(stanfordcorenlp) as tokenize:
        train = read_other_dataset(
            tokenize, datasets[0], shrink=shrink)
        if len(datasets) == 2:
            test = read_other_dataset(
                tokenize, datasets[1], shrink=shrink)
        else:
            numpy.random.seed(seed)
            alldata = numpy.random.permutation(train)
            train = alldata[:-len(alldata) // 10]
            test = alldata[-len(alldata) // 10:]

    print('constract vocabulary based on frequency')
    vocab = make_vocab(train)

    if word_emb is not None:
        print('load word embedding')
        emb, vocab = load_glove_vocab(word_emb, vocab=vocab.keys())
    else:
        emb = None

    train = transform_to_array(train, vocab)
    test = transform_to_array(test, vocab)

    return train, test, vocab, emb


def load_glove_vocab(path, vocab, max_vocab=None):
    vocab = set(vocab)
    if path.endswith('.zip'):
        @contextmanager
        def load(path):
            zf = ZipFile(path)
            fin = zf.open(zf.namelist()[0], 'r')
            yield io.TextIOWrapper(fin)
            fin.close()
            zf.close()
    else:
        load = open
    new_vocab = {}
    new_vocab['<eos>'] = 0
    new_vocab['<unk>'] = 1
    emb = []
    with load(path) as fin:
        for line in fin:
            vals = line.strip().split(' ')
            word = vals[0]
            if word in vocab and word not in new_vocab:
                emb.append(list(map(numpy.float32, vals[1:])))
                new_vocab[word] = len(new_vocab)
            if max_vocab is not None and len(new_vocab) == max_vocab:
                break
    special_tokens = numpy.random.uniform(-0.01, 0.01, size=(2, len(emb[0]))).tolist()
    emb = numpy.array(special_tokens + emb, dtype=numpy.float32)
    return emb, new_vocab


def load_glove(path, max_vocab=None, vocab=None):
    if vocab is not None:
        return load_glove_vocab(path, vocab, max_vocab=max_vocab)
    arr = pd.read_csv(path, sep=' ', header=None, nrows=max_vocab)
    vocab = {}
    vocab['<eos>'] = 0
    vocab['<unk>'] = 1
    del_inds = []
    for i in range(len(arr)):
        a = arr.iloc[i, 0]
        if a in vocab:
            del_inds.append(i)
        else:
            vocab[a] = i + 2 - len(del_inds)
    emb = arr.iloc[:, 1:].values.astype(numpy.float32)
    emb = numpy.delete(emb, del_inds, axis=0)
    emb = numpy.vstack(
        (numpy.random.uniform(-0.01, 0.01, size=(2, emb.shape[1])),
         emb))
    return emb, vocab
