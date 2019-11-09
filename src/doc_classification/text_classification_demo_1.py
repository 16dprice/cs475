# following the tutorial at the following:
# https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a

import numpy as np

# load the training data
# will load the test data later
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)


# print(twenty_train.target_names) # prints all the categories
# print("\n".join(twenty_train.data[0].split("\n")[:3])) # prints first line of the first data file


# this is a way of counting term frequency
# ultimately, this is poor because it gives more weight to longer documents (long doc = more words)
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
# print(X_train_counts.shape)


# this will do a tf-idf (term frequency times inverse document frequency) calculation
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape) # (n_samples, n_features)



# Time for some classification!
########################################################################################################################

############################################## Naive Bayes Classification ##############################################


# there are many variants of NB, but this is the one that the tutorial uses
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
text_clf_nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# this trains the classifier
text_clf_nb = text_clf_nb.fit(twenty_train.data, twenty_train.target)

# now to test the classifier
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
predicted_nb = text_clf_nb.predict(twenty_test.data)
print(np.mean(predicted_nb == twenty_test.target))


############################################ SVM (Support Vector Machines) #############################################


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
])

text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)

predicted_svm = text_clf_svm.predict(twenty_test.data)
print(np.mean(predicted_svm == twenty_test.target))


########################################### Grid Search (Parameter Tuning) #############################################


from sklearn.model_selection import GridSearchCV
# create a list of parameters we would like to tune and define what those paramaters can be
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

# this screws up my computer for some reason
# TODO: fix it
# gs_clf = GridSearchCV(text_clf_nb, parameters, n_jobs=-1) # n_jobs = -1 => use all cores available
# gs_clf.fit(twenty_train.data, twenty_train.target)
#
# print(gs_clf.best_score_)
# print(gs_clf.best_params_)