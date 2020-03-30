# word_vectors = api.load("glove-wiki-gigaword-100")
#
# print(word_vectors['king'] - word_vectors['man'] + word_vectors['woman'])
# print(word_vectors['queen'])
# print(word_vectors['king'] - word_vectors['man'] + word_vectors['woman'] - word_vectors['queen'])

# https://gist.github.com/bhaettasch/d7f4e22e79df3c8b6c20

import gensim
model = gensim.models.KeyedVectors.load_word2vec_format('./data/en/GoogleNews-vectors-negative300.bin', binary=True)

print(model.wv['king'])
print(model.wv['queen'])
print(model.wv['man'])
print(model.wv['woman'])
