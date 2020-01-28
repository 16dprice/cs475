import gensim

f = open("./datasetSentences.txt")
train_corpus = []

for i, line in enumerate(f):
    if i > 0:
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i - 1]))

model = gensim.models.doc2vec.Doc2Vec(vector_size=400, min_count=2, epochs=100)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

model.save("imdb_model.model")