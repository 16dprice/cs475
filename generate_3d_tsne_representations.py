#! /usr/local/Cellar/python/3.7.3/bin/python3

"""
see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html to learn more about the
to learn more about the arguments passed via command line
"""

import gensim
import os
import collections
import smart_open
import random
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import sys
from doc_classification import svm_classifier as svm_clf

# early_exaggeration = float(sys.argv[1])  # default: 12.0
#
# doc_vector_size = int(sys.argv[2])
# num_epochs = int(sys.argv[3])
#
# date_made = str(sys.argv[4])
early_exaggeration = 30.0
doc_vector_size = 30
num_epochs = 30
date_made = "nov15_19"

train_file = os.getcwd() + '/src/aggregate_train_corpus.txt'

# yield produces a value that can only be iterated over once
# that is, it's not stored in memory and is deallocated once it's iterated over
# so yielding, instead of returning, speeds things up
# https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/
def read_corpus(fname, tokens_only=False):

    with smart_open.open(fname, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # for training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus(train_file))

model_path = "./src/doc2vec_models/{}/3d_representations/vs_{}_epochs_{}".format(date_made, doc_vector_size, num_epochs)
model_name = model_path + "/aggregate_model.model"

if not os.path.exists(model_path):
    os.mkdir(model_path)

if os.path.exists(model_name):
    model = gensim.models.doc2vec.Doc2Vec.load(model_name)
else:
    model = gensim.models.doc2vec.Doc2Vec(vector_size=doc_vector_size, min_count=2, epochs=num_epochs)
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save(model_name)

doc_vectors = np.array(model.docvecs.vectors_docs)
embedded_doc_vectors = TSNE(n_components=3, early_exaggeration=early_exaggeration, random_state=1).fit_transform(doc_vectors)

# training data gets labeled in order of being presented
# so, in particular, the following training categories will be given the commented labels
svm_classifier = svm_clf.SVMClassifier(os.getcwd())
svm_classifier.train([
    '/src/mtg_articles.txt', # label 0
    '/src/sports_articles.txt', # label 1
    '/src/bharatanatyam_pdfs.txt' # label 2
])

# print(svm_classifier.predict(' '.join(train_corpus[1].words)))

# init the list so there's not a lot of appending when adding the labels
# the label is hardcoded
write_file = open(model_path + "/3d_coords.csv", "w")
embedded_doc_vectors_with_words = [None for i in range(len(doc_vectors))]
for doc_id in range(len(doc_vectors)):
    # 0: mtg
    # 1: sports
    # 2: bharatanatyam
    write_file.write("{},{},{},{}\n".format(
        embedded_doc_vectors[doc_id, 0],
        embedded_doc_vectors[doc_id, 1],
        embedded_doc_vectors[doc_id, 2],
        svm_classifier.predict(' '.join(train_corpus[doc_id].words))
    ))

