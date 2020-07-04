import os

import gensim
from src.ProjectCorpus import ProjectCorpus

class TrainDoc2VecModel:

    def __init__(self):
        self.save_dir = "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/"

    def get_model(self, train_corpus, save_path, vector_size, epochs, train_new_model=False):

        model_path = self.save_dir + save_path

        if os.path.exists(model_path) and not train_new_model: return gensim.models.doc2vec.Doc2Vec.load(model_path)

        model = gensim.models.doc2vec.Doc2Vec(vector_size=vector_size, min_count=2, epochs=epochs)

        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.save(model_path)

        return model

    def get_aggregate_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_aggregate_corpus()
        save_path = "aggregate/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_mtg_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_mtg_corpus()
        save_path = "mtg/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_sports_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_sports_corpus()
        save_path = "sports/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_dance_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_dance_corpus()
        save_path = "dance/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_mtg_and_sports_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_mtg_and_sports_corpus()
        save_path = "mtg_and_sports/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_20news_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_news20_corpus()
        save_path = "20news/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model

    def get_old_aggregate_model(self, vector_size, epochs, train_new_model=False):

        corpus = ProjectCorpus()
        train_corpus = corpus.get_old_aggregate_corpus()
        save_path = "aggregate_old/models/vs_{}_epochs_{}.model".format(vector_size, epochs)

        model = self.get_model(train_corpus, save_path, vector_size, epochs, train_new_model)

        return model
