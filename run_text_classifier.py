#!/usr/bin/env python
import argparse
import json
import os
import sys

import chainer

import nets
import nlp_utils


def setup_model(args):
    sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
    setup = json.load(
        open(os.path.join(args.model_setup, 'args.json')))
    sys.stderr.write(json.dumps(setup, indent=2) + '\n')

    vocab = json.load(
        open(os.path.join(args.model_setup, 'vocab.json')))
    n_class = setup['n_class']

    # Setup a model
    if setup['model'] == 'hier':
        model = nets.SWEMhier(
            n_class, n_vocab=len(vocab), emb_size=setup['emb_size'],
            n_units=setup['unit'], dropout=setup['dropout'],
            window=setup['window'])
    elif setup['model'] == 'concat':
        model = nets.SWEMconcat(
            n_class, n_vocab=len(vocab), emb_size=setup['emb_size'],
            n_units=setup['unit'], dropout=setup['dropout'])

    chainer.serializers.load_npz(
        os.path.join(args.model_setup, 'best_model.npz'), model)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    elif chainer.backends.intel64.is_ideep_available():
        setattr(chainer.config, 'use_ideep', 'auto')
        model.to_intel64()

    return model, vocab, setup


def run_online(gpu, tokenize):
    # predict labels online
    for l in sys.stdin:
        l = l.strip()
        if not l:
            print('# blank line')
            continue
        text = nlp_utils.normalize_text(l)
        words = tokenize(text)
        xs = nlp_utils.transform_to_array([words], vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            prob = model.predict(xs, softmax=True)[0]
        answer = int(model.xp.argmax(prob))
        score = float(prob[answer])
        print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))


def run_batch(gpu, tokenize,  batchsize=64):
    # predict labels by batch

    def predict_batch(words_batch):
        xs = nlp_utils.transform_to_array(words_batch, vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            probs = model.predict(xs, softmax=True)
        answers = model.xp.argmax(probs, axis=1)
        scores = probs[model.xp.arange(answers.size), answers].tolist()
        for words, answer, score in zip(words_batch, answers, scores):
            print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))

    batch = []
    for l in sys.stdin:
        l = l.strip()
        if not l:
            if batch:
                predict_batch(batch)
                batch = []
            print('# blank line')
            continue
        text = nlp_utils.normalize_text(l)
        words = tokenize(text)
        batch.append(words)
        if len(batch) >= batchsize:
            predict_batch(batch)
            batch = []
    if batch:
        predict_batch(batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', default='result/',
                        help='Model setup dictionary.')
    args = parser.parse_args()

    model, vocab, setup = setup_model(args)
    with nlp_utils.get_tokenizer(setup['char_based'], setup['stanfordcorenlp']) as tokenize:
        if args.gpu >= 0:
            run_batch(args.gpu, tokenize)
        else:
            run_online(args.gpu, tokenize)
