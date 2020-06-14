import gensim
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import os

for model_name in os.listdir("./src/doc2vec_models/summer_2020/mtg_and_sports/models"):

    model_path = "./src/doc2vec_models/summer_2020/mtg_and_sports/models/{}".format(model_name)
    model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    X = np.array(model.docvecs.vectors_docs)

    kmeans_fit_predict = KMeans(n_clusters=2, random_state=0).fit_predict(X)

    class_counts = Counter(kmeans_fit_predict)

    if 50 in class_counts.values() and 38 in class_counts.values():
        print("Success. Model tested was {}".format(model_name))
    else:
        print(
            "Error. {} in class 0 and {} in class 1. Model tested was {}"
                .format(class_counts[0], class_counts[1], model_name)
        )
