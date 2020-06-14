import gensim
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import os

for model_name in os.listdir("./src/doc2vec_models/summer_2020/aggregate/models"):

    model_path = "./src/doc2vec_models/summer_2020/aggregate/models/{}".format(model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    X = np.array(model.docvecs.vectors_docs)

    kmeans_fit_predict = KMeans(n_clusters=3, random_state=0).fit_predict(X)

    class_counts = Counter(kmeans_fit_predict)
    class_tallies = class_counts.values()

    if 50 in class_tallies and 38 in class_tallies and 14 in class_tallies:
        print("Success. Model tested was {}".format(model_name))
    else:
        print(
            "Error. {} in class 0. {} in class 1. {} in class 2. Model tested was {}"
                .format(class_counts[0], class_counts[1], class_counts[2], model_name)
        )
