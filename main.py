import os
import time as t

early_exaggeration = 12.0
doc_vector_size = 10
epochs = 40

delta_exagg = 2.0
delta_vs = 5
delta_epochs = 10

while early_exaggeration <= 30.0:
    while epochs <= 100:
        while doc_vector_size <= 50:
            start = t.time()
            os.system("./generate_tsne_representations.py {} {} {}".format(early_exaggeration, doc_vector_size, epochs))
            end = t.time()

            print("Done with ee {} vs {} epochs {} in {} seconds".format(early_exaggeration, doc_vector_size, epochs, end - start))
            doc_vector_size += delta_vs

        doc_vector_size = 10
        epochs += delta_epochs

    epochs = 10
    early_exaggeration += delta_exagg

