#! /usr/local/Cellar/python/3.7.3/bin/python3

"""
see https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html to learn more about the
to learn more about the arguments passed via command line
"""

import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from src.ProjectCorpus import ProjectCorpus
from src.TrainDoc2VecModel import TrainDoc2VecModel

early_exaggeration = 24.0
doc_vector_size = 15
num_epochs = 100

# train_corpus = list(read_corpus(train_file))
projectCorpus = ProjectCorpus()
train_corpus = list(projectCorpus.get_mtg_and_sports_corpus())

model_path = "./src/doc2vec_models/summer_2020/mtg_and_sports/models/"

model_trainer = TrainDoc2VecModel()
model = model_trainer.get_mtg_and_sports_model(doc_vector_size, num_epochs)

doc_vectors = np.array(model.docvecs.vectors_docs)
embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=1).fit_transform(doc_vectors)

# init the list so there's not a lot of appending when adding the labels
# the label is hardcoded
embedded_doc_vectors_with_words = [None for i in range(len(doc_vectors))]
for doc_id in range(len(doc_vectors)):
    # 0 - 49: mtg
    # 50 - 87: sports
    # 88+: bharatanatyam
    if doc_id <= 49:
        label = "m"
        color = dict(facecolor='blue', alpha=0.5)
    else:
        label = "s"
        color = dict(facecolor='red', alpha=0.5)

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

# file formats supported are png, pdf, and some mores. jpg and jpeg are not supported
plt.savefig(
    "./src/doc2vec_models/summer_2020/mtg_and_sports/tsne_plots/vs_{}_epochs_{}_ee_{}.png".format(doc_vector_size, num_epochs, early_exaggeration)
)
plt.show()
