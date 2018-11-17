#!/usr/bin/env python
import argparse
import datetime
import json
import os

import chainer
from chainer import training
from chainer.training import extensions

import nets
from nlp_utils import convert_seq
import text_datasets


def main():
    current_datetime = '{}'.format(datetime.datetime.today())
    parser = argparse.ArgumentParser(description='Train SWEM model')
    # Default hyperparameters as specified in the author's implementation
    # https://github.com/dinghanshen/SWEM/blob/master/eval_dbpedia_emb.py
    parser.add_argument('--batchsize', '-b', type=int, default=50,
                        help='Number of images in each mini-batch')
    # author's impl was using average cross entropy loss instead of sum
    parser.add_argument('--lr', type=float, default=4e-6,
                        help='Learning rate')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--word-emb', type=str, default=None,
                        help='Pretrained Glove file')
    parser.add_argument('--window', type=int, default=153,
                        help='Pooling window size for SWEM')
    parser.add_argument('--dataset', '-data', default='dbpedia',
                        choices=['dbpedia', 'imdb.binary', 'imdb.fine',
                                 'TREC', 'stsa.binary', 'stsa.fine',
                                 'custrev', 'mpqa', 'rt-polarity', 'subj'],
                        help='Name of dataset.')
    parser.add_argument('--model', '-model', default='concat',
                        choices=['concat', 'hier'])

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))

    if args.word_emb is not None:
        initial_emb, vocab = text_datasets.load_glove(args.word_emb)
        emb_size = initial_emb.shape[1]
    else:
        initial_emb = None
        vocab = None
        emb_size = args.unit

    # Load a dataset
    if args.dataset == 'dbpedia':
        train, test, vocab = text_datasets.get_dbpedia(vocab=vocab)
    elif args.dataset.startswith('imdb.'):
        train, test, vocab = text_datasets.get_imdb(
            vocab=vocab, fine_grained=args.dataset.endswith('.fine'))
    elif args.dataset in ['TREC', 'stsa.binary', 'stsa.fine',
                          'custrev', 'mpqa', 'rt-polarity', 'subj']:
        train, test, vocab = text_datasets.get_other_text_dataset(
            args.dataset, vocab=vocab)

    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # Setup a model
    if args.model == 'hier':
        model = nets.SWEMhier(
            n_class, n_vocab=len(vocab), emb_size=emb_size,
            n_units=args.unit, dropout=args.dropout,
            initial_emb=initial_emb, window=args.window)
    elif args.model == 'concat':
        model = nets.SWEMconcat(
            n_class, n_vocab=len(vocab), emb_size=emb_size,
            n_units=args.unit, dropout=args.dropout,
            initial_emb=initial_emb)

    classifier = chainer.links.Classifier(model, label_key='ys')

    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        classifier.to_gpu()  # Copy the model to the GPU
    elif chainer.backends.intel64.is_ideep_available():
        setattr(chainer.config, 'use_ideep', 'auto')
        classifier.to_intel64()

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(classifier)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        converter=convert_seq, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, classifier,
        converter=convert_seq, device=args.gpu))

    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    # Save vocabulary and model's setting
    if not os.path.isdir(args.out):
        os.mkdir(args.out)
    current = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(current, args.out, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(current, args.out, 'best_model.npz')
    model_setup = args.__dict__
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(args.out, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
