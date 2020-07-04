from src.TrainDoc2VecModel import TrainDoc2VecModel

model_trainer = TrainDoc2VecModel()

aggregate_list = [
    {"vs": 15, "e": 30},
    {"vs": 20, "e": 40},
    {"vs": 10, "e": 50},
    {"vs": 15, "e": 80}
]

news20_list = [
    {"vs": 15, "e": 30},
    {"vs": 20, "e": 40},
    {"vs": 10, "e": 50},
    {"vs": 15, "e": 80}
]

for aggregate in aggregate_list:
    file = open("./csv_models_for_uta/aggregate_vs_{}_epochs_{}.csv".format(aggregate["vs"], aggregate["e"]), "w")
    model = model_trainer.get_aggregate_model(aggregate["vs"], aggregate["e"])
    for vec in model.docvecs.vectors_docs:
        file.write(','.join(map(str, vec)))
        file.write('\n')
    file.close()

for news20 in news20_list:
    file = open("./csv_models_for_uta/news20_vs_{}_epochs_{}.csv".format(news20["vs"], news20["e"]), "w")
    model = model_trainer.get_20news_model(news20["vs"], news20["e"])
    for vec in model.docvecs.vectors_docs:
        file.write(','.join(map(str, vec)))
        file.write('\n')
    file.close()
