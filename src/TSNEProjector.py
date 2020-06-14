import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class TSNEProjector:

    def project_aggregate_model(self, model, early_exaggeration):

        doc_vectors = np.array(model.docvecs.vectors_docs)
        embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=1).fit_transform(doc_vectors)

        embedded_doc_vectors_with_words = [None for _ in range(len(doc_vectors))]
        for doc_id in range(len(doc_vectors)):
            if doc_id <= 49:
                label = "m"
                color = dict(facecolor='blue', alpha=0.5)
            else:
                label = "s"
                color = dict(facecolor='red', alpha=0.5)

            embedded_doc_vectors_with_words[doc_id] = [
                (embedded_doc_vectors[doc_id, 0], embedded_doc_vectors[doc_id, 1]), label, color
            ]

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
            "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/mtg_and_sports/tsne_plots/vs_{}_epochs_{}_ee_{}.png".format(
                len(doc_vectors[0]), model.epochs, early_exaggeration)
        )
        plt.show()
