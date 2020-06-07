# https://gist.github.com/bhaettasch/d7f4e22e79df3c8b6c20

import gensim
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

word_vectors = gensim.models.KeyedVectors.load_word2vec_format('./data/en/GoogleNews-vectors-negative300.bin', binary=True)

# countries_to_capitals = dict()
#
# countries_to_capitals['China'] = 'Beijing'
# countries_to_capitals['Russia'] = 'Moscow'
# countries_to_capitals['Japan'] = 'Tokyo'
# countries_to_capitals['Turkey'] = 'Ankara'
# countries_to_capitals['Poland'] = 'Warsaw'
# countries_to_capitals['Germany'] = 'Berlin'
# countries_to_capitals['France'] = 'Paris'
# countries_to_capitals['Italy'] = 'Rome'
# countries_to_capitals['Greece'] = 'Athens'
# countries_to_capitals['Spain'] = 'Madrid'
# countries_to_capitals['Portugal'] = 'Lisbon'

# for country in countries_to_capitals:
#     print('{} has capital {}'.format(country, countries_to_capitals[country]))
#
#     country_file = open('./data/capital_country_files/{}.txt'.format(country), 'w+')
#     capital_file = open('./data/capital_country_files/{}.txt'.format(countries_to_capitals[country]), 'w+')
#
#     for num in word_vectors[country]:
#         country_file.write(str(num) + "\n")
#     country_file.close()
#
#     for num in word_vectors[countries_to_capitals[country]]:
#         capital_file.write(str(num) + "\n")
#     capital_file.close()

early_exaggeration = 16.0

word_vectors = np.array(word_vectors.vectors)
embedded_doc_vectors = TSNE(n_components=2, early_exaggeration=early_exaggeration, random_state=1).fit_transform(word_vectors)

# find the bounds for the axes
minX = min(embedded_doc_vectors[0:, 0])
maxX = max(embedded_doc_vectors[0:, 0])

minY = min(embedded_doc_vectors[0:, 1])
maxY = max(embedded_doc_vectors[0:, 1])

x_offset = 0.1 * (maxX - minX)
y_offset = 0.1 * (maxY - minY)

# define axes and label each point
plt.axis([minX - x_offset, maxX + x_offset, minY - y_offset, maxY + y_offset])
for point in embedded_doc_vectors:
    plt.text(point[0], point[1])

plt.show()
