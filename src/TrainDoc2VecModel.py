import gensim
from src.ProjectCorpus import ProjectCorpus

class TrainDoc2VecModel:

    def get_aggregate_model(self, vector_size, epochs):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_aggregate_corpus()

        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)

        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        model.save("/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/vs_{}_epochs_{}.model".format(vector_size, epochs))

        return model
