from src.KMeansPredictor import KMeansPredictor
from src.TrainDoc2VecModel import TrainDoc2VecModel

# vs=15 epochs=80

vector_size = 10
epochs = 10

model_trainer = TrainDoc2VecModel()
# model = model_trainer.get_aggregate_model(vector_size=15, epochs=80)

while vector_size <= 50:
    while epochs <= 100:
        model = model_trainer.get_aggregate_model(
            vector_size=vector_size,
            epochs=epochs
        )

        kmeans_accuracy = KMeansPredictor.get_kmeans_accuracy_normalized(
            n_clusters=3,
            model=model,
            expectations=[50, 38, 14],
            num_passes=100
        )

        is_perfect_text = ""
        if kmeans_accuracy == 1.0:
            is_perfect_text = "Perfect!"

        if kmeans_accuracy >= 0.95:
            print("Model with vs={} epochs={} {}".format(vector_size, epochs, is_perfect_text))

        epochs += 10

    vector_size += 5
    epochs = 10
