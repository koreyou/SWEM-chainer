# SWEM: Simple Word-Embedding-Based Models

This project implements [Shen et al. 2018. Baseline Needs More Love: On Simple Word-Embedding-Based Models and Associated Pooling Mechanisms. ACL.](https://aclanthology.info/papers/P18-1041/p18-1041) in [Chainer](https://chainer.org/).

Dataset is switchable among below:
- [DBPedia Ontology dataset](https://github.com/zhangxiangxiao/Crepe) (dbpedia): Predict its ontology class from the abstract of an Wikipedia article.
- [IMDB Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (imdb.binary, imdb.fine): Predict its sentiment from a review about a movie. `.binary`'s classes are positive/negative. `.fine`'s classes are ratings [0-1]/[2-3]/[7-8]/[9-10].
- [TREC Question Classification](http://cogcomp.cs.illinois.edu/Data/QA/QC/) (TREC): Predict the type of its answer from a factoid question.
- [Stanford Sentiment Treebank](https://nlp.stanford.edu/sentiment/index.html) (stsa.binary, stsa.fine): Predict its sentiment from a review about a movie. `.binary`'s classes are positive/negative. `.fine`'s classes are [negative]/[somewhat negative]/[neutral]/[somewhat positive]/[positive].
- [Customer Review Datasets](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html) (custrev): Predict its sentiment (positive/negative) from a review about a product.
- [MPQA Opinion Corpus](http://www.cs.pitt.edu/mpqa/) (mpqa): Predict its opinion polarity from a phrase.
- [Scale Movie Review Dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (rt-polarity): Predict its sentiment (positive/negative) from a review about a movie.
- [Subjectivity datasets](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (subj): Predict subjectivity (subjective/objective) from a sentnece about a movie.

Some of datasets are downloaded from @harvardnlp's [repository](https://github.com/harvardnlp/sent-conv-torch/tree/master/data). Thank you.

# How to Run

## Basics

To train a model:  
```
python train_text_classifier.py
```

It will download the dataset and start training the model.
Note that **it does not use pretrained word embedding by default**.

The output directory `result` contains:  
- `best_model.npz`: a model snapshot, which won the best accuracy for validation data during training
- `vocab.json`: model's vocabulary dictionary as a json file
- `args.json`: model's setup as a json file, which also contains paths of the model and vocabulary

To apply the saved model to your sentences, feed the sentences through stdin:  
```
cat sentences_to_be_classifed.txt | python run_text_classifier.py -g 0 --model-setup result/args.json
```
The classification result is given by stdout.

## Using other resources (recommended)

To use pretrained word embedding feed the path via `--word-emb` option.
It currently only support GloVe format (either zipped or raw text).
To use word2vec text format, just delete the first line. (untested)

You may also want to use the same tokenizer as pretrained GloVe.
To do so, download StanfordCoreNLP from https://stanfordnlp.github.io/CoreNLP/ , unzip the data to any directory and specify the CoreNLP root directory via `--stanfordcorenlp` option.


See `python train_text_classifier.py -h` for full options.



# Reproducing the paper

I ran the model on DBpedia Ontology classification task (with git commit 3596301d8861a7d0191df2c20eb5e96e998f2382).
The hyperparameters follows the author's implementation[^1].
I used 300d GloVe vectors pretrained on Common Crawl (840B tokens, 2.2M vocab, cased) from https://nlp.stanford.edu/projects/glove/ .

[^1]: https://github.com/dinghanshen/SWEM

```
python train_text_classifier.py \
    -g 0 \
    --model concat \
    --word-emb ./glove.840B.300d.zip \
    --stanfordcorenlp ./stanford-corenlp-full-2018-10-05
```
(for "concat" model)


| Model            | Accuracy |
|------------------|----------|
| Bag-of-means[^2] | 90.45    |
| CNN[^2]          | 98.28    |
| LSTM[^2]         | 98.55    |
| fastText[^2]     | 98.10    |
| SWEM-concat[^2]  | 98.57    |
| SWEM-hier[^2]    | 98.54    |
| SWEM-concat      | 98.49    |
| SWEM-hier        | 98.59    |


[^2]: https://aclanthology.info/papers/P18-1041/p18-1041

Bottom two is my implementation and others are extracted from the paper[^2].
You can see that my implementation (roughly) reproduces the results from the original paper.

> On a K80 GPU machine, training roughly takes about 3 minutes each epoch and 5 epochs for Debpedia to converge, ...

(From the author's Github page[^1])

My implementation took about 4 minutes each epoch on K80 GPU (Google Colaboratory) and 8 epochs to converge.
I belive difference is negligiable considering that we are using different frameworks and computation environments.

## Note

It is crucial that number of UNKs (words that were not in the vocabulary) is kept low.
The performance is wrecked when the UNK ratio is high.
Try keeping UNK ratio low by using reasonable choice of pretrained word embedding and using tokenizer that matches your word embedding.
In my experiment, unk ratio was 3.6%.

# LICENSE

Most of the code is derived from [Chainer](https://github.com/chainer/chainer) examples, thus it should be treated as MIT license.
The code I added (e.g. nets.py, check diff for the detail) should be treated as [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/).
