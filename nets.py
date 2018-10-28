import numpy

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import reporter

embed_init = chainer.initializers.Uniform(.25)


def block_embed(embed, x, dropout=0.):
    """Embedding function followed by convolution

    Args:
        embed (callable): A :func:`~chainer.functions.embed_id` function
            or :class:`~chainer.links.EmbedID` link.
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`): Input variable, which
            is a :math:`(B, L)`-shaped int array. Its first dimension
            :math:`(B)` is assumed to be the *minibatch dimension*.
            The second dimension :math:`(L)` is the length of padded
            sentences.
        dropout (float): Dropout ratio.

    Returns:
        ~chainer.Variable: Output variable. A float array with shape
        of :math:`(B, N, L, 1)`. :math:`(N)` is the number of dimensions
        of word embedding.

    """
    e = embed(x)
    e = F.dropout(e, ratio=dropout)
    e = F.transpose(e, (0, 2, 1))
    e = e[:, :, :, None]
    return e


class SWEMBase(chainer.Chain):

    """The base class for SWEM (Simple Word-Embedding-based Models)

    This model embed tokens to word embedding, encode embedding to
    with pooling (which needs to be implemented in derived classes)
    and applies two layer MLP.

    Args:
        n_class (int): The number of classes to be predicted.
        n_vocab (int): The size of vocabulary.
        emb_size (int): The number of units word embedding.
        n_units (int): The number of units of MLP.
        dropout (float): The dropout ratio.
    """

    def __init__(self, n_class, n_vocab, emb_size, n_units, window,
                 dropout=0.2, initial_emb=None):
        super(SWEMBase, self).__init__()
        if initial_emb is None:
            initial_emb = embed_init
        with self.init_scope():
            self.embed = L.EmbedID(
                n_vocab, emb_size, ignore_label=-1, initialW=initial_emb)
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(n_units, n_class)
        self.dropout = dropout
        self.window = window

    def forward(self, xs):
        return self.predict(xs)

    def predict(self, xs, softmax=False, argmax=False):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        x_len = [len(x) for x in xs]
        z = self.encode(ex_block, x_len)
        h = F.relu(self.l1(F.dropout(z, self.dropout)))
        logits = self.l2(F.dropout(h, self.dropout))
        if softmax:
            return F.softmax(logits).array
        elif argmax:
            return self.xp.argmax(logits.array, axis=1)
        else:
            return logits

    def encode(self, ex_block, x_len):
        raise NotImplementedError()


class SWEMhier(SWEMBase):

    def encode(self, ex_block, x_len):
        if ex_block.shape[2] > self.window:
            ex_block = F.max_pooling_2d(ex_block, [self.window, 1], stride=1)
        return F.max(F.squeeze(ex_block, -1), axis=2)


class SWEMconcat(SWEMBase):

    def encode(self, ex_block, x_len):
        emb_ave = F.sum(F.squeeze(ex_block, -1), axis=2) / self.xp.array(x_len)[:, None]
        if ex_block.shape[2] > self.window:
            # no need for pooling when length is smaller than the window
            ex_block = F.max_pooling_2d(ex_block, [self.window, 1], stride=1)
        emb_hier = F.max(F.squeeze(ex_block, -1), axis=2)
        return F.concat((emb_hier, emb_ave), axis=1)
