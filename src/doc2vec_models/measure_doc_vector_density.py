import gensim
import statistics


def l1_norm(v):
    return sum(map(lambda x: abs(x), v))


def l2_norm(v):
    return sum(map(lambda x: x ** 2, v)) ** 0.5


# 0 is dense
# 1 is sparse
def sparseness(v):
    return ( (len(v) ** 0.5) - (l1_norm(v) / l2_norm(v)) ) / ( (len(v) ** 0.5) - 1 )


def avg_sparseness(vectors):
    return statistics.mean(map(sparseness, vectors))


def median_sparseness(vectors):
    return statistics.median(map(sparseness, vectors))


# min/max vector size and delta vector size
min_vs = 10
max_vs = 50
d_vs = 5

# min/max epochs and delta epochs
min_epochs = 10
max_epochs = 100
d_epochs = 10

avg_density_file = open("avg_doc_vector_density.txt", "w+")
median_density_file = open("median_doc_vector_density.txt", "w+")

for vs in range(min_vs, max_vs + d_vs, d_vs):
    for epochs in range(min_epochs, max_epochs + d_epochs, d_epochs):
        model = gensim.models.doc2vec.Doc2Vec.load("vs_{}_epochs_{}/aggregate_model.model".format(vs, epochs))

        doc_vectors = model.docvecs.vectors_docs

        # I should know these values already, but just to be sure in case something got saved improperly
        vector_size = model.docvecs.vectors_docs.shape[1]
        model_epochs = model.epochs

        avg_sparseness_of_model = avg_sparseness(doc_vectors)
        median_sparseness_of_model = median_sparseness(doc_vectors)

        avg_density_file.write("{},{},{}\n".format(vector_size, model_epochs, avg_sparseness_of_model))
        median_density_file.write("{},{},{}\n".format(vector_size, model_epochs, median_sparseness_of_model))


avg_density_file.close()
median_density_file.close()
