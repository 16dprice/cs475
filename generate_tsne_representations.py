#! /usr/local/Cellar/python/3.7.3/bin/python3

"""
see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html to learn more about the
to learn more about the arguments passed via command line
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.TrainDoc2VecModel import TrainDoc2VecModel

early_exaggeration = 24.0
model_trainer = TrainDoc2VecModel()

def train_model(doc_vector_size, num_epochs):

    model = model_trainer.get_aggregate_model(doc_vector_size, num_epochs, True)

    doc_vectors = np.array(model.docvecs.vectors_docs)
    embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=1).fit_transform(doc_vectors)

    # init the list so there's not a lot of appending when adding the labels
    # the label is hardcoded
    embedded_doc_vectors_with_words = [None for _ in range(len(doc_vectors))]
    for doc_id in range(len(doc_vectors)):
        # 0 - 49: mtg
        # 50 - 87: sports
        # 88+: bharatanatyam
        label = doc_id
        if doc_id <= 49:
            # label = "m"
            color = dict(facecolor='blue', alpha=0.5)
        elif 50 <= doc_id < 88:
            # label = "s"
            color = dict(facecolor='red', alpha=0.5)
        else:
            # label = "b"
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

    plt.savefig("/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/aggregate/tsne_plots/vs_{}_epochs_{}_ee_{}.png".format(doc_vector_size, num_epochs, early_exaggeration))
    plt.show()

import time as t

vector_size = 5
epochs = 10

delta_vs = 5
delta_epochs = 10

while epochs <= 100:
    while vector_size <= 50:
        start = t.time()
        train_model(vector_size, epochs)
        end = t.time()

        print("Done with vs {} epochs {} in {} seconds".format(vector_size, epochs, end - start))
        vector_size += delta_vs

    epochs += delta_epochs
    vector_size = 5
