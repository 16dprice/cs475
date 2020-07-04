import gensim
import numpy as np
from src.ProjectCorpus import ProjectCorpus
import matplotlib.pyplot as plt

vector_size = 3
epochs = 20

corpus = ProjectCorpus()
train_corpus = corpus.get_small_corpus()
save_dir = "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020"
save_path = "{}/small_corpus/models/vs_{}_epochs_{}.model".format(save_dir, vector_size, epochs)

model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)

model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
model.save(save_path)

doc_vectors = np.array(model.docvecs.vectors_docs)

embedded_doc_vectors_with_words = [None for _ in range(len(doc_vectors))]
for doc_id in range(len(doc_vectors)):
    if doc_id <= 9:
        label = "m"
        color = dict(facecolor='blue', alpha=0.5)
    else:
        label = "s"
        color = dict(facecolor='red', alpha=0.5)

    embedded_doc_vectors_with_words[doc_id] = [
        (doc_vectors[doc_id, 0], doc_vectors[doc_id, 1]), label, color
    ]

# find the bounds for the axes
minX = min(doc_vectors[0:, 0])
maxX = max(doc_vectors[0:, 0])

minY = min(doc_vectors[0:, 1])
maxY = max(doc_vectors[0:, 1])

x_offset = 0.1 * (maxX - minX)
y_offset = 0.1 * (maxY - minY)

# define axes and label each point
plt.axis([minX - x_offset, maxX + x_offset, minY - y_offset, maxY + y_offset])
for point, label, color in embedded_doc_vectors_with_words:
    plt.text(point[0], point[1], label, bbox=color)

# file formats supported are png, pdf, and some mores. jpg and jpeg are not supported
plt.savefig(
    "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/small_corpus/plots/vs_{}_epochs_{}.png".format(
        len(doc_vectors[0]), model.epochs)
)
plt.show()