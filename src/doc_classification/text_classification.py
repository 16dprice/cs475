import os
import smart_open

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

train_file = os.getcwd() + '/../aggregate_train_corpus.txt'

file_contents = []
with smart_open.open(train_file, encoding="UTF-8") as f:
    for i, line in enumerate(f):
        file_contents.append(line)

# should be 35 mtg, 14 sports, 10 dance
train_data = file_contents[:35] + file_contents[50:64] + file_contents[70:80]
# 15 mtg, 6 sports, 4 dance
test_data = file_contents[35:50] + file_contents[64:70] + file_contents[80:]

train_targets = []
for i in range(35): train_targets.append(0)
for i in range(14): train_targets.append(1)
for i in range(10): train_targets.append(2)

test_targets = []
for i in range(15): test_targets.append(0)
for i in range(6 ): test_targets.append(1)
for i in range(4 ): test_targets.append(2)

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()


############################################## Naive Bayes Classification ##############################################


# there are many variants of NB, but this is the one that the tutorial uses
text_clf_nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

text_clf_nb.fit(train_data, train_targets)

predicted_nb = text_clf_nb.predict(test_data)
print(np.mean(predicted_nb == test_targets))


############################################ SVM (Support Vector Machines) #############################################


text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
])

text_clf_svm.fit(train_data, train_targets)

predicted_svm = text_clf_svm.predict(test_data)
print(np.mean(predicted_svm == test_targets))
