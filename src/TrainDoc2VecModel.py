import gensim
from src.ProjectCorpus import ProjectCorpus

class TrainDoc2VecModel:

    def __init__(self):
        self.save_dir = "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/"

    def get_model(self, train_corpus, save_path, vector_size, epochs):

        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)

        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(save_path)

        return model

    def get_aggregate_model(self, vector_size, epochs):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_aggregate_corpus()
        save_path = "{}/aggregate/vs_{}_epochs_{}.model".format(self.save_dir, vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs)

        return model

    def get_mtg_model(self, vector_size, epochs):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_mtg_corpus()
        save_path = "{}/mtg/vs_{}_epochs_{}.model".format(self.save_dir, vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs)

        return model

    def get_sports_model(self, vector_size, epochs):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_sports_corpus()
        save_path = "{}/sports/vs_{}_epochs_{}.model".format(self.save_dir, vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs)

        return model

    def get_dance_model(self, vector_size, epochs):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_dance_corpus()
        save_path = "{}/dance/vs_{}_epochs_{}.model".format(self.save_dir, vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs)

        return model
