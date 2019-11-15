import smart_open
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

class SVMClassifier:

    def __init__(self, init_path):
        self.train_data = []
        self.test_data = []

        self.train_targets = []
        self.test_targets = []

        self.init_path = init_path

        self.text_clf_svm = None

    @staticmethod
    def get_file_contents(path):
        file_contents = []
        with smart_open.open(path, encoding="UTF-8") as f:
            for i, line in enumerate(f):
                file_contents.append(line)

        return file_contents

    def add_files(self, rel_paths):
        target = 0
        for path in rel_paths:
            file_contents = self.get_file_contents(self.init_path + path)

            num_train_contents = round(0.7 * len(file_contents))
            num_test_contents = len(file_contents) - num_train_contents

            self.train_data += file_contents[:num_train_contents]
            self.test_data += file_contents[num_train_contents:]

            for i in range(num_train_contents): self.train_targets.append(target)
            for i in range(num_test_contents ): self.test_targets.append(target)

            target += 1

    def train(self, rel_paths):
        self.add_files(rel_paths)

        self.text_clf_svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
        ])

        self.text_clf_svm.fit(self.train_data, self.train_targets)

    def accuracy(self):
        if self.text_clf_svm is None: return -1
        return np.mean(self.text_clf_svm.predict(self.test_data) == self.test_targets)

