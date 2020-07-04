#! /usr/local/Cellar/python/3.7.3/bin/python3

from src.TSNEProjector import TSNEProjector
from src.TrainDoc2VecModel import TrainDoc2VecModel

save_path = "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/aggregate"

model_trainer = TrainDoc2VecModel()
projector = TSNEProjector()

import time as t

init_vector_size = 5

vector_size = init_vector_size
epochs = 10

delta_vs = 5
delta_epochs = 10

max_vector_size = 25
max_epochs = 100

while epochs <= max_epochs:
    while vector_size <= max_vector_size:
        start = t.time()
        model = model_trainer.get_aggregate_model(vector_size, epochs, True)
        projector.project_model(model, 24.0, save_path)
        end = t.time()

        print("Done with vs {} epochs {} in {} seconds".format(vector_size, epochs, end - start))
        vector_size += delta_vs

    vector_size = init_vector_size
    epochs += delta_epochs
