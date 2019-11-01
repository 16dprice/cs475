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

doc_vector_size = int(sys.argv[2])
num_epochs = int(sys.argv[3])

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

model_path = "./src/doc2vec_models/vs_{}_epochs_{}".format(doc_vector_size, num_epochs)
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
embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=1).fit_transform(doc_vectors)

# init the list so there's not a lot of appending when adding the labels
# the label is hardcoded
embedded_doc_vectors_with_words = [None for i in range(len(doc_vectors))]
for doc_id in range(len(doc_vectors)):
    # 0 - 49: mtg
    # 50 - 68: sports
    # 69+: bharatanatyam
    if doc_id <= 49:
        label = "m"
        color = dict(facecolor='blue', alpha=0.5)
    elif 50 <= doc_id <= 68:
        label = "s"
        color = dict(facecolor='red', alpha=0.5)
    else:
        label = "b"
        color = dict(facecolor='black', alpha=0.5)

    embedded_doc_vectors_with_words[doc_id] = [(embedded_doc_vectors[doc_id, 0], embedded_doc_vectors[doc_id, 1]), label, color]

# find the bounds for the axes
minX = min(embedded_doc_vectors[0:, 0])
maxX = max(embedded_doc_vectors[0:, 0])

minY = min(embedded_doc_vectors[0:, 1])
maxY = max(embedded_doc_vectors[0:, 1])

x_offset = 0.1 * (maxX - minX)
y_offset = 0.1 * (maxY - minY)

# define axes and label each point
plt.axis([minX - x_offset, maxX + x_offset, minY - y_offset, maxY + y_offset])
for point, label, color in embedded_doc_vectors_with_words:
    plt.text(point[0], point[1], label, bbox=color)

plt.title("Early Exaggeration: {} -- Vector Size: {} -- Epochs: {}".format(early_exaggeration, doc_vector_size, num_epochs))

# file formats supported are png, pdf, and some mores. jpg and jpeg are not supported
plt.savefig(model_path + "/ee_{}.png".format(early_exaggeration))
