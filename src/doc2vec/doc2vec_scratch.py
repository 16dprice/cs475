# Following the tutorial from https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb
# most code is pulled straight from that site

import gensim
import os
import collections
import smart_open
import random

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

# because of the relatively small number of training examples, the number of epochs is relatively high
# the vocab is a bunch of information about words in the document
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# a neat example of inferring a paragraph vector
# print(model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires']))

# assess the model
ranks = []
second_ranks = []

# this is more of a 'sanity check' rather than a real accuracy test
# this basically just checks to see if the model acts as we expect
# that is, documents should be very closely related to themselves
# if they're not, something went terribly wrong
# TODO: check which ones are populating the second_ranks array (look at which documents aren't ranking 'sanely')
for doc_id in range(len(train_corpus)):

    # infer a vector from something that we already know about
    # if the model is good, this should be extremely closely related to itself in the model
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)

    # returns a list of form [(int, float), ... ] where the int is the doc_id and
    # the float is cosine similarity
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # the smaller the index of the doc_id, the better the model is
    # so, a rank of 0 is very good
    # this would mean that the model predicted the document itself as the closest document
    rank = [docid for docid, sim in sims].index(doc_id)

    # if this is a list of 0's, the model has done very well
    ranks.append(rank)

    # the second closest document
    second_ranks.append(sims[1])

# an easy way to see if the model is 'sane'
# should expect a large number of 0's to be present in ranks
# print(collections.Counter(ranks))

# pick a random document from the corpus and infer a vector from the model
doc_id = random.randint(0, len(train_corpus) - 1)

# sometimes this is pretty bad because a document can be 'isolated'
# that is, the model may be trained well, but there does not exist a document very related to the chosen document at all
print('Train Document ({}): <<{}>>\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id, sim_dist = second_ranks[doc_id]  # the second most similar doc
print('Similar Document ({}, {}): <<{}>>\n'.format(sim_id, sim_dist, ' '.join(train_corpus[sim_id].words)))

model.save("doc2vec/new_model.model")
