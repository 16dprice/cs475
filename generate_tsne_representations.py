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

early_exaggeration = float(sys.argv[1])  # default: 12.0
random_state = int(sys.argv[2])  # default: None

mtg_train_file = os.getcwd() + '/src/aggregate_train_corpus.txt'

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


train_corpus = list(read_corpus(mtg_train_file))
model = gensim.models.doc2vec.Doc2Vec.load("./src/aggregate_model.model")

doc_vectors = np.array(model.docvecs.vectors_docs)
embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=random_state).fit_transform(doc_vectors)

embedded_doc_vectors_with_words = [None for i in range(len(doc_vectors))]
for doc_id in range(len(doc_vectors)):
    label = "s"
    if doc_id < 50:
        label = "m"

    embedded_doc_vectors_with_words[doc_id] = [(embedded_doc_vectors[doc_id, 0], embedded_doc_vectors[doc_id, 1]), label]


minX = min(embedded_doc_vectors[0:, 0])
maxX = max(embedded_doc_vectors[0:, 0])

minY = min(embedded_doc_vectors[0:, 1])
maxY = max(embedded_doc_vectors[0:, 1])

plt.axis([minX - 10, maxX + 10, minY - 10, maxY + 10])
for point, label in embedded_doc_vectors_with_words:
    plt.text(point[0], point[1], label)

plt.title("Learning Exaggeration: {} -- Random State: {}".format(early_exaggeration, random_state))

if not os.path.exists("./tsne_plot_outputs/random_state_{}".format(random_state)):
    os.mkdir("./tsne_plot_outputs/random_state_{}".format(random_state))

plt.savefig("./tsne_plot_outputs/random_state_{}/ee_{}_rs_{}.png".format(random_state, early_exaggeration, random_state))
