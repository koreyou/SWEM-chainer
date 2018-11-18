import chainer
from chainer.dataset.convert import concat_examples
from chainer.cuda import to_cpu
import numpy as np


def predict_batch(model, iterator, converter=concat_examples,
                  device=None, label_key=-1):
    def predict_batch(*args, **kwargs):
        if isinstance(label_key, int):
            if not (-len(args) <= label_key < len(args)):
                msg = 'Label key %d is out of bounds' % label_key
                raise ValueError(msg)
            t = args[label_key]
            if label_key == -1:
                args = args[:-1]
            else:
                args = args[:label_key] + args[label_key + 1:]
        elif isinstance(label_key, str):
            if label_key not in kwargs:
                msg = 'Label key "%s" is not found' % label_key
                raise ValueError(msg)
            t = kwargs[label_key]
            del kwargs[label_key]
        kwargs['softmax'] = True
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            probs = model.predict(*args, **kwargs)
        return to_cpu(probs), to_cpu(t)

    probs_all = []
    labels_all = []
    for batch in iterator:
        in_arrays = converter(batch, device)
        if isinstance(in_arrays, tuple):
            optimizer.update(*in_arrays)
            probs, t = predict_batch(*in_arrays)
        elif isinstance(in_arrays, dict):
            probs, t = predict_batch(**in_arrays)
        else:
            probs, t = predict_batch(in_arrays)

        probs_all.append(probs)
        labels_all.append(t)
    probs_all = np.concatenate(probs_all, axis=0)
    preds_all = np.argmax(probs_all, axis=1)
    labels_all = np.concatenate(labels_all)
    return probs_all, preds_all, labels_all
    
        
def evaluate(model, dataset, converter=concat_examples, device=None,
             batchsize=64, label_key=-1):
    iterator = chainer.iterators.SerialIterator(
        dataset, batchsize, repeat=False, shuffle=False)
    probs, preds, labels = predict_batch(
        model, iterator, converter=converter, device=device,
        label_key=label_key
    )
    accuracy = np.average(preds == labels)
    print('Accuracy: %f' % accuracy)

