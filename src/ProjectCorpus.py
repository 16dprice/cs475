import gensim
import smart_open

# DataIndices gives the starting and ending index for data in a file
# It is non-inclusive (to make it work better with range function)
from collections import namedtuple
DataIndices = namedtuple("DataIndices", "start end")

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


class ProjectCorpus:

    def __init__(self):
        self.mtg_articles_path = "/home/dj/PycharmProjects/cs475/src/data/mtg_articles.txt"
        self.sports_articles_path = "/home/dj/PycharmProjects/cs475/src/data/sports_articles.txt"
        self.dance_articles_path = "/home/dj/PycharmProjects/cs475/src/data/bharatanatyam_pdfs.txt"
        self.aggregate_corpus_path = "/home/dj/PycharmProjects/cs475/src/data/aggregate_train_corpus.txt"
        self.mtg_and_sports_articles_path = "/home/dj/PycharmProjects/cs475/src/data/mtg_and_sports_articles.txt"
        self.small_corpus_path = "/home/dj/PycharmProjects/cs475/src/data/small_train_corpus.txt"
        self.news20_path = "/home/dj/PycharmProjects/cs475/src/data/20news_corpus.txt"
        self.old_aggregate_corpus_path = "/home/dj/PycharmProjects/cs475/src/data/aggregate_train_corpus_old.txt"

    def get_mtg_corpus(self): return read_corpus(self.mtg_articles_path)
    def get_sports_corpus(self): return read_corpus(self.sports_articles_path)
    def get_dance_corpus(self): return read_corpus(self.dance_articles_path)
    def get_aggregate_corpus(self): return read_corpus(self.aggregate_corpus_path)
    def get_mtg_and_sports_corpus(self): return read_corpus(self.mtg_and_sports_articles_path)
    def get_small_corpus(self): return read_corpus(self.small_corpus_path)
    def get_news20_corpus(self): return read_corpus(self.news20_path)
    def get_old_aggregate_corpus(self): return read_corpus(self.old_aggregate_corpus_path)

    @staticmethod
    def get_mtg_data_indices(): return DataIndices(0, 50)

    @staticmethod
    def get_sports_data_indices(): return DataIndices(50, 88)

    @staticmethod
    def get_dance_data_indices(): return DataIndices(88, 103)
