from src.ExperimentModel import ExperimentModel
from src.ProjectCorpus import ProjectCorpus
from src.TrainDoc2VecModel import TrainDoc2VecModel

corpus_obj = ProjectCorpus()
corpus_path = corpus_obj.old_aggregate_corpus_path

model_trainer = TrainDoc2VecModel()
model = model_trainer.get_old_aggregate_model(15, 100)

class_ranges = [
    ExperimentModel.create_class_range(0, 50),
    ExperimentModel.create_class_range(50, 88),
    ExperimentModel.create_class_range(88, 102)
]
experiment_model = ExperimentModel(corpus_path, model, class_ranges)

print(experiment_model.svm_classify())
experiment_model.generate_tsne_representation(12.0, "/home/dj/PycharmProjects/cs475/src/doc2vec_models/summer_2020/aggregate_old")
