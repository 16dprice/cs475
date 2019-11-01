import os
import smart_open
import gensim
import statistics
import multiprocessing as mp


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

min_vs = 10
max_vs = 50
d_vs = 5

min_e = 10
max_e = 100
d_e = 10


# this part can be done in parallel
def train_model(vs, e, count):
    model = gensim.models.doc2vec.Doc2Vec(vector_size=vs, min_count=2, epochs=e)
    model.build_vocab(train_corpus)

    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.save("./models/vs_{}_epochs_{}_{}.model".format(vs, e, count))

    print("Done with vs {} e {} on iteration {}".format(vs, e, count))


pool = mp.Pool(processes=mp.cpu_count())

for i in range(10, 50):
    for vector_size in range(min_vs, max_vs + d_vs, d_vs):
        for epochs in range(min_e, max_e + d_e, d_e):
            pool.apply_async(train_model, args=(vector_size, epochs, i))

pool.close()
pool.join()


avg_density_file = open("./models/avg_doc_vector_density.txt", "a+")
median_density_file = open("./models/median_doc_vector_density.txt", "a+")

# do this part serially
for i in range(10, 50):
    for vs in range(min_vs, max_vs + d_vs, d_vs):
        for e in range(min_e, max_e + d_e, d_e):

            model = gensim.models.doc2vec.Doc2Vec.load("./models/vs_{}_epochs_{}_{}.model".format(vs, e, i))

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
