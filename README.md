** WIP: This project is still incomplete. **

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

To train a model:  
```
python train_text_classifier.py --word-emb glove.6B.100d.txt
```
The output directory `result` contains:  
- `best_model.npz`: a model snapshot, which won the best accuracy for validation data during training
- `vocab.json`: model's vocabulary dictionary as a json file
- `args.json`: model's setup as a json file, which also contains paths of the model and vocabulary


To apply the saved model to your sentences, feed the sentences through stdin:  
```
cat sentences_to_be_classifed.txt | python run_text_classifier.py -g 0 --model-setup result/args.json
```
The classification result is given by stdout.


# LICENSE

Most of the code is derived from [Chainer](https://github.com/chainer/chainer) examples, thus it should be treated as MIT license.
The code I added (e.g. nets.py, check diff for the detail) should be treated as [CC0](https://creativecommons.org/share-your-work/public-domain/cc0/).
