import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import gensim

model = gensim.models.doc2vec.Doc2Vec.load("src/aggregate_model.model")
doc_vectors = np.array(model.docvecs.vectors_docs)
embedded_doc_vectors = TSNE(n_components=2).fit_transform(doc_vectors)

print(embedded_doc_vectors)
