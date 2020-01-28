import smart_open
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

class SVMClassifier:

    def __init__(self, init_path):
        self.train_data = []
        self.test_data = []

        self.train_targets = []
        self.test_targets = []

        self.init_path = init_path

        self.text_clf_svm = None

    @staticmethod
    def get_file_contents(path):
        file_contents = []
        with smart_open.open(path, encoding="UTF-8") as f:
            for i, line in enumerate(f):
                file_contents.append(line)

        return file_contents

    def add_files(self, rel_paths):
        target = 0
        for path in rel_paths:
            file_contents = self.get_file_contents(self.init_path + path)

            num_train_contents = round(0.7 * len(file_contents))
            num_test_contents = len(file_contents) - num_train_contents

            self.train_data += file_contents[:num_train_contents]
            self.test_data += file_contents[num_train_contents:]

            for i in range(num_train_contents): self.train_targets.append(target)
            for i in range(num_test_contents ): self.test_targets.append(target)

            target += 1

    def train(self, rel_paths):
        self.add_files(rel_paths)

        self.text_clf_svm = Pipeline([
            ('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter_no_change=5, random_state=42))
        ])

        self.text_clf_svm.fit(self.train_data, self.train_targets)

    def accuracy(self):
        if self.text_clf_svm is None: return -1
        return np.mean(self.text_clf_svm.predict(self.test_data) == self.test_targets)

    def predict(self, text):
        if self.text_clf_svm is None: return -1
        return self.text_clf_svm.predict([text])[0]

# svm_classifier = SVMClassifier(os.getcwd())
# svm_classifier.train([
#     '/../mtg_articles.txt',
#     '/../sports_articles.txt',
#     '/../bharatanatyam_pdfs.txt'
# ])
#
# print(svm_classifier.accuracy())
# print(svm_classifier.predict('welcome back everyone todays best of three videos will be temur elementals it not your traditional list though we are also playing flood of tears and omniscience for combo finish in case the game goes long or you just turbo ramp with your risen reefs if you like elementals or just want to play another flood of tears deck you should check this one out time stamps match match match match match match temur elementals core set standard ali aintrazi creatures cavalier of thorns hydroid krasis leafkin druid llanowar elves omnath locus of the roil risen reef planeswalkers tamiyo collector of tales instants growth spiral sorceries lava coil flood of tears enchantments omniscience lands forest rootbound crag steam vents temple of mystery breeding pool hinterland harbor stomping ground sideboard aether gust lava coil cindervines flame sweep shifting ceratops chandra awakened inferno buy this deck export text format export arena format again you re essentially temur elementals deck with combo finish the combo being omniscience in play flood of tears in hand or in your graveyard and having tamiyo collector of tales this will allow you to cast flood of tears putting omniscience into play then playing tamiyo and regrowing flood of tears then play whatever else you have in your hand and rinse and repeat you ll have infinite draw with risen reef and infinite damage with omnath locus of the roil when you bring in chandra awakened inferno you are also able to make infinite chandra emblems we do all this couple of times in the video if you are confused about how it works or want to see it in action that it for today hope you have wonderful week and if you play any of these decks and you enjoy them tweet at me love hearing about people enjoying my decks at fnm or wherever you may take them as always thanks for reading ali aintrazi follow me alieldrazi twitch channel think twice mtg podcast'))