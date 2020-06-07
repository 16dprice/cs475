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
        self.mtg_articles_path = "data/mtg_articles.txt"
        self.sports_articles_path = "data/sports_articles.txt"
        self.dance_articles_path = "data/bharatanatyam_pdfs.txt"
        self.aggregate_corpus_path = "data/aggregate_train_corpus.txt"

    def get_mtg_corpus(self): read_corpus(self.mtg_articles_path)
    def get_sports_corpus(self): read_corpus(self.sports_articles_path)
    def get_dance_corpus(self): read_corpus(self.dance_articles_path)
    def get_aggregate_corpus(self): read_corpus(self.aggregate_corpus_path)

    @staticmethod
    def get_mtg_data_indices(): DataIndices(0, 50)

    @staticmethod
    def get_sports_data_indices(): DataIndices(50, 88)

    @staticmethod
    def get_dance_data_indices(): DataIndices(88, 103)
