from src.KMeansPredictor import KMeansPredictor
from src.TrainDoc2VecModel import TrainDoc2VecModel

vector_size = 5
epochs = 10

model_trainer = TrainDoc2VecModel()

while vector_size <= 50:
    while epochs <= 100:
        model = model_trainer.get_20news_model(
            vector_size=vector_size,
            epochs=epochs
        )

        predictor = KMeansPredictor(
            n_clusters=2,
            random_state=1,
            model=model
        )
        prediction = predictor.fit_predict()

        class_tallies = predictor.get_class_tallies(prediction)
        print("{} {} {}".format(sorted(class_tallies), vector_size, epochs))

        # kmeans_accuracy = KMeansPredictor.get_kmeans_accuracy(
        #     n_clusters=3,
        #     model=model,
        #     expectations=[50, 38, 26],
        #     num_passes=100
        # )
        #
        # is_perfect_text = ""
        # if kmeans_accuracy == 1.0:
        #     is_perfect_text = "Perfect!"
        #
        # if kmeans_accuracy >= 0.95:
        #     print("Model with vs={} epochs={} {}".format(vector_size, epochs, is_perfect_text))

        epochs += 10

    vector_size += 5
    epochs = 10
