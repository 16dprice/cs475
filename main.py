import os

early_exaggeration = 12.0
random_state = 2

delta_exagg = 2.0
delta_rs = 1

while early_exaggeration <= 50.0:
    while random_state <= 10:
        os.system("./generate_tsne_representations.py {} {}".format(early_exaggeration, random_state))
        print("Done with ee {} rs {}".format(early_exaggeration, random_state))
        random_state += delta_rs
    random_state = 2
    early_exaggeration += delta_exagg