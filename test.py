#! /usr/local/Cellar/python/3.7.3/bin/python3

from src.TSNEProjector import TSNEProjector
from src.TrainDoc2VecModel import TrainDoc2VecModel

early_exaggeration = 24.0
doc_vector_size = 15
num_epochs = 80

model_trainer = TrainDoc2VecModel()
model = model_trainer.get_mtg_and_sports_model(doc_vector_size, num_epochs)

projector = TSNEProjector()
projector.project_aggregate_model(model, early_exaggeration)
