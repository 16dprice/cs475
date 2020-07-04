import smart_open
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from collections import namedtuple
import numpy as np

from src.TSNEProjector import TSNEProjector

ClassRange = namedtuple("ClassRange", "start end")

def get_file_contents(path):
    file_contents = []
    with smart_open.open(path, encoding="UTF-8") as f:
        for i, line in enumerate(f):
            file_contents.append(line)
    return file_contents

class ExperimentModel:

    def __init__(self, corpus_path, model, class_ranges):
        self.corpus_path = corpus_path
        self.model = model
        self.class_ranges = class_ranges

    @staticmethod
    def create_class_range(start, end):
        return ClassRange(start, end)

    def generate_tsne_representation(self, early_exaggeration, save_path):
        projector = TSNEProjector()
        projector.project_model(self.model, early_exaggeration, save_path)

    def svm_classify(self):

        train_data, train_targets, test_data, test_targets = [], [], [], []
        corpus = get_file_contents(self.corpus_path)
        target = 0

        for class_range in self.class_ranges:
            start = class_range.start
            end = class_range.end
            train_num = round(0.7 * (end - start))
            test_num = (end - start) - train_num

            train_data += corpus[start : start + train_num]
            test_data += corpus[start + train_num : end]

            for i in range(train_num): train_targets.append(target)
            for i in range(test_num): test_targets.append(target)

            target += 1


        text_clf_svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
        ])

        text_clf_svm.fit(train_data, train_targets)

        predicted_svm = text_clf_svm.predict(test_data)
        return np.mean(predicted_svm == test_targets)