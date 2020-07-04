from sklearn.cluster import KMeans
from src.TrainDoc2VecModel import TrainDoc2VecModel
from collections import Counter
import numpy as np

# TODO: figure out how to find the documents that are in which cluster (so that I can see which docs are being misclassified)

class KMeansPredictor:

    def __init__(self, n_clusters, random_state, model, expectations=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = model
        self.class_tally_expectations = expectations

    def set_class_tally_expectations(self, expectations):
        self.class_tally_expectations = expectations

    def fit_predict(self):
        return KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        ).fit_predict(np.array(self.model.docvecs.vectors_docs))

    def fit_predict_normalized(self):
        X = self.model.docvecs.vectors_docs
        length = np.sqrt((X**2).sum(axis=1))[:, None]
        X = X / length

        return KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state
        ).fit_predict(X)

    def get_class_tallies(self, prediction):
        class_counts = Counter(prediction)
        class_tallies = class_counts.values()

        return class_tallies

    def model_predicts_all_correct(self, prediction):
        model_predicts_correctly = True
        class_tallies = self.get_class_tallies(prediction)

        for expectation in self.class_tally_expectations:
            if expectation not in class_tallies:
                model_predicts_correctly = False

        return model_predicts_correctly

    @staticmethod
    def get_kmeans_accuracy(n_clusters, model, expectations, num_passes):
        successes = 0
        for rand_state in range(num_passes):
            predictor = KMeansPredictor(
                n_clusters=n_clusters,
                random_state=rand_state,
                model=model,
                expectations=expectations
            )
            if predictor.model_predicts_all_correct(predictor.fit_predict()): successes += 1

        return successes / num_passes

    @staticmethod
    def get_kmeans_accuracy_normalized(n_clusters, model, expectations, num_passes):
        successes = 0
        for rand_state in range(num_passes):
            predictor = KMeansPredictor(
                n_clusters=n_clusters,
                random_state=rand_state,
                model=model,
                expectations=expectations
            )
            if predictor.model_predicts_all_correct(predictor.fit_predict_normalized()): successes += 1

        return successes / num_passes