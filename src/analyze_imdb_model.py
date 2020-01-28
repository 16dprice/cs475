import gensim
import numpy as np
import scipy

f = open("./datasetSentences.txt")
train_corpus = []

for i, line in enumerate(f):
    if i > 0:
        train_corpus.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i - 1]))

max_index = 11854

# 11829 - "Terrible ."
# 11833 - "An opportunity missed ."
# 2725 - "Insanely hilarious !"
# 2485 - "... a solid , well-formed satire ."
# analogy_index = np.random.randint(0, max_index + 1)
positive_analogy_index = 2485
negative_analogy_index = 11829

# 66 - "Disney has always been hit-or-miss when bringing beloved kids ' books to the screen ... Tuck Everlasting is a little of both ."
# 6560 - "the makers of mothman prophecies succeed in producing that most frightening of all movies mediocre horror film too bad to be good and too good to be bad"
# original_sentence_index = np.random.randint(0, max_index + 1)
original_sentence_index = 66


model = gensim.models.doc2vec.Doc2Vec.load("./imdb_model.model")

# print(model.docvecs[positive_analogy_index])
# print(model.docvecs[negative_analogy_index])
# print(scipy.spatial.distance.cosine(model.docvecs[positive_analogy_index], model.docvecs[negative_analogy_index]))
# quit()

prediction = model.docvecs[original_sentence_index] - model.docvecs[negative_analogy_index] + model.docvecs[positive_analogy_index]
most_similar_to_prediction = model.docvecs.most_similar([prediction], topn=5)

print("Original Sentence Index: {}. Original Sentence: \'{}\'\n".format(original_sentence_index, ' '.join(train_corpus[original_sentence_index].words)))
print("Negative Analogy Sentence Index: {}. Negative Analogy Sentence: \'{}\'".format(negative_analogy_index, ' '.join(train_corpus[negative_analogy_index].words)))
print("Positive Analogy Sentence Index: {}. Positive Analogy Sentence: \'{}\'\n".format(positive_analogy_index, ' '.join(train_corpus[positive_analogy_index].words)))
for index, sim in most_similar_to_prediction:
    print("Index: {}. Similarity: {}. Sentence: \'{}\'".format(index, round(sim, 3), ' '.join(train_corpus[index].words)))

print("")
most_similar_to_original = model.docvecs.most_similar([model.docvecs[original_sentence_index]], topn=5)
for index, sim in most_similar_to_original:
    print("Index: {}. Similarity: {}. Sentence: \'{}\'".format(index, round(sim, 3), ' '.join(train_corpus[index].words)))
