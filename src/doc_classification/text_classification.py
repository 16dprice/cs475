import os
import smart_open

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

def get_file_contents(path):
    file_contents = []
    with smart_open.open(path, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            file_contents.append(line)

    return file_contents


######################################### Splitting Data (70% train, 30% test) #########################################


mtg_articles_file_contents = get_file_contents(os.getcwd() + '/../mtg_articles.txt')
sports_articles_file_contents = get_file_contents(os.getcwd() + '/../sports_articles.txt')
bharatanatyam_pdfs_file_contents = get_file_contents(os.getcwd() + '/../bharatanatyam_pdfs.txt')

num_train_mtg_articles = round(0.7 * len(mtg_articles_file_contents))
num_train_sports_articles = round(0.7 * len(sports_articles_file_contents))
num_train_bharatanatyam_pdfs = round(0.7 * len(bharatanatyam_pdfs_file_contents))

num_test_mtg_articles = len(mtg_articles_file_contents) - num_train_mtg_articles
num_test_sports_articles = len(sports_articles_file_contents) - num_train_sports_articles
num_test_bharatanatyam_pdfs = len(bharatanatyam_pdfs_file_contents) - num_train_bharatanatyam_pdfs

# should be 35 mtg, 27 sports, 10 dance
train_data = []
train_data += mtg_articles_file_contents[:num_train_mtg_articles]
train_data += sports_articles_file_contents[:num_train_sports_articles]
train_data += bharatanatyam_pdfs_file_contents[:num_train_bharatanatyam_pdfs]

# 15 mtg, 11 sports, 4 dance
test_data = []
test_data += mtg_articles_file_contents[num_train_mtg_articles:]
test_data += sports_articles_file_contents[num_train_sports_articles:]
test_data += bharatanatyam_pdfs_file_contents[num_train_bharatanatyam_pdfs:]


train_targets = []
for i in range(num_train_mtg_articles):       train_targets.append(0)
for i in range(num_train_sports_articles):    train_targets.append(1)
for i in range(num_train_bharatanatyam_pdfs): train_targets.append(2)

test_targets = []
for i in range(num_test_mtg_articles):        test_targets.append(0)
for i in range(num_test_sports_articles):     test_targets.append(1)
for i in range(num_test_bharatanatyam_pdfs):  test_targets.append(2)

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
# originally I got 0.65 on this
# after adding 20 more sports articles (5 of each type) this goes up to 0.866


############################################ SVM (Support Vector Machines) #############################################


text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
])

text_clf_svm.fit(train_data, train_targets)

predicted_svm = text_clf_svm.predict(test_data)
print(np.mean(predicted_svm == test_targets))
# originally I got 0.95 on this, which is fantastic
# after adding 20 more sports articles (5 of each type) this goes up to 1.0, which is even more fantastic