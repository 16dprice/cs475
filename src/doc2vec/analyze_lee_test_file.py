import gensim
import os
import smart_open

# set file names for train and test data
test_data_dir = '{}'.format(os.sep).join([gensim.__path__[0], 'test', 'test_data'])

lee_train_file = test_data_dir + os.sep + 'lee_background.cor'
lee_test_file = test_data_dir + os.sep + 'lee.cor'

# yield produces a value that can only be iterated over once
# that is, it's not stored in memory and is deallocated once it's iterated over
# so yielding, instead of returning, speeds things up
# https://pythontips.com/2013/09/29/the-python-yield-keyword-explained/
def read_corpus(fname, tokens_only=False):

    with smart_open.open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # for training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


# define the train and test corpus based on the train and test file obtained from gensim
train_corpus = list(read_corpus(lee_train_file))
test_corpus = list(read_corpus(lee_test_file, tokens_only=True))

for line in train_corpus:
    print line

