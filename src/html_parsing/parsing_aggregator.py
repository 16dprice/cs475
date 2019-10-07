from html_parsing.cool_stuff_inc_article_parser import CoolStuffIncArticleParser
from html_parsing.espn_news_wire_parser import EspnNewsWireParser
import gensim


class ParsingAggregator:

    def __init__(self):
        self.cool_stuff_inc_urls = [
    # Jim Davis
    "https://www.coolstuffinc.com/a/jimdavis-09202019-love-hates-for-throne-of-eldraine",
    "https://www.coolstuffinc.com/a/jimdavis-09162019-sending-the-cats-to-the-astrolabe",
    "https://www.coolstuffinc.com/a/jimdavis-09132019-force-of-negation-is-the-most-important-card-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-09092019-taking-stoneforge-mystic-in-a-darker-direction-with-stoneforge-pox",
    "https://www.coolstuffinc.com/a/jimdavis-09062019-faithless-looting-replacements-updating-dredge-izzet-phoenix-and-mardu-pyromancer",
    "https://www.coolstuffinc.com/a/jimdavis-09022019-casting-armageddon-in-standard",
    "https://www.coolstuffinc.com/a/jimdavis-08272019-banning-faithless-looting-in-modern-was-a-mistake",
    "https://www.coolstuffinc.com/a/jimdavis-08262019-breaking-standard-with-four-color-kethis-combo",
    "https://www.coolstuffinc.com/a/jimdavis-08232019-exploring-different-ways-to-play-on-a-digital-playground",
    "https://www.coolstuffinc.com/a/jimdavis-08192019-be-your-own-best-friend-with-chandra-tribal",
    "https://www.coolstuffinc.com/a/jimdavis-08162019-the-future-of-magic-esports-and-organized-play",
    "https://www.coolstuffinc.com/a/jimdavis-08122019-they-said-snow",
    "https://www.coolstuffinc.com/a/jimdavis-08092019-bored-of-standard-and-modern-try-these-brews",
    "https://www.coolstuffinc.com/a/jimdavis-08052019-reanimating-drakuseth-to-burninate-standard",
    "https://www.coolstuffinc.com/a/jimdavis-08022019-mc-prediction-results-and-goblins-sideboard-guide",
    "https://www.coolstuffinc.com/a/jimdavis-08012019-playing-goblins-at-the-mythic-championship",
    "https://www.coolstuffinc.com/a/jimdavis-07252019-five-mythic-championship-iv-predictions",
    "https://www.coolstuffinc.com/a/jimdavis-07222019-playing-hogaak-fair-in-modern-with-hogaak-pox",
    "https://www.coolstuffinc.com/a/jimdavis-07192019-the-verdict-on-cube-cards-from-core-set-2020-and-modern-horizons",
    "https://www.coolstuffinc.com/a/jimdavis-07152019-modern-vial-goblins",
    "https://www.coolstuffinc.com/a/jimdavis-07122019-combos-to-watch-for-in-week-one-of-core-set-2020-standard",
    "https://www.coolstuffinc.com/a/jimdavis-07082019-fate-shift-combo-in-standard",
    "https://www.coolstuffinc.com/a/jimdavis-07052019-ten-new-core-set-2020-standard-brews",
    "https://www.coolstuffinc.com/a/jimdavis-07012019-winning-on-turn-three-with-blister-burn",
    "https://www.coolstuffinc.com/a/jimdavis-06282019-love-hate-for-core-set-2020",
    "https://www.coolstuffinc.com/a/jimdavis-06242019-go-ninja-go-ninja-go",
    "https://www.coolstuffinc.com/a/jimdavis-06212019-first-look-at-core-set-2020",
    "https://www.coolstuffinc.com/a/jimdavis-06172019-arcades-alert-when-walls-attack",
    "https://www.coolstuffinc.com/a/jimdavis-06142019-splinter-twin-is-back-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-06112019-starting-with-an-idea-modern-four-color-saheeli-combo",
    "https://www.coolstuffinc.com/a/jimdavis-06072019-the-biggest-trends-in-standard-and-modern-for-scgcon",
    "https://www.coolstuffinc.com/a/jimdavis-06032019-putting-the-big-pig-to-work-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-05312019-the-complete-standard-four-color-command-the-dreadhorde-guide",
    "https://www.coolstuffinc.com/a/jimdavis-05272019-trying-to-punish-those-with-no-basics",
    "https://www.coolstuffinc.com/a/jimdavis-05242019-five-cards-that-are-secretly-excellent-in-standard-right-now",
    "https://www.coolstuffinc.com/a/jimdavis-05202019-a-force-to-be-reckoned-with",
    "https://www.coolstuffinc.com/a/jimdavis-05172019-four-standard-superfriends-brews",
    "https://www.coolstuffinc.com/a/jimdavis-05132019-killing-people-on-turn-two-with-neoform",
    "https://www.coolstuffinc.com/a/jimdavis-05102019-war-of-the-spark-week-one-planeswalker-report-card",
    "https://www.coolstuffinc.com/a/jimdavis-05062019-ahhhh-gruul-monsters",
    # Ali Aintrazi
    "https://www.coolstuffinc.com/a/aliaintrazi-09182019-watch-it-burn-preview-for-throne-of-eldraine",
    "https://www.coolstuffinc.com/a/aliaintrazi-09112019-best-of-one-golos-in-new-standard",
    "https://www.coolstuffinc.com/a/aliaintrazi-09062019-jund-dinosaurs-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-09042019-omniscience-draft-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-08302019-fandom-legends-1st-place-five-color-golos-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-08282019-battle-of-one-grixis-artifacts",
    "https://www.coolstuffinc.com/a/aliaintrazi-08212019-battle-of-one-big-blue",
    "https://www.coolstuffinc.com/a/aliaintrazi-08142019-battle-of-one-kykars-angels",
    "https://www.coolstuffinc.com/a/aliaintrazi-08092019-standard-temur-elementals",
    "https://www.coolstuffinc.com/a/aliaintrazi-08072019-battle-of-one-orzhov-aristocrats"
]
        self.espn_news_wire_urls = [
            # source page : http://www.espn.com/espn/wire/_/sportId/wire?sportId=10
            # MLB (baseball)
            "http://www.espn.com/espn/wire?section=mlb&id=27789479",
            "http://www.espn.com/espn/wire?section=mlb&id=27788752",
            "http://www.espn.com/espn/wire?section=mlb&id=27789171",
            "http://www.espn.com/espn/wire?section=mlb&id=27789163",
            "http://www.espn.com/espn/wire?section=mlb&id=27788560",
            # NFL (football)
            "http://www.espn.com/espn/wire?section=nfl&id=27790463",
            "http://www.espn.com/espn/wire?section=nfl&id=27789599",
            "http://www.espn.com/espn/wire?section=nfl&id=27789527",
            "http://www.espn.com/espn/wire?section=nfl&id=27789066",
            "http://www.espn.com/espn/wire?section=nfl&id=27788730",
            # NBA (basketball)
            "http://www.espn.com/espn/wire?section=nba&id=27788802",
            "http://www.espn.com/espn/wire?section=nba&id=27780769",
            "http://www.espn.com/espn/wire?section=nba&id=27777276",
            "http://www.espn.com/espn/wire?section=nba&id=27773865",
            # NHL (hockey)
            "http://www.espn.com/espn/wire?section=nhl&id=27788185",
            "http://www.espn.com/espn/wire?section=nhl&id=27788143",
            "http://www.espn.com/espn/wire?section=nhl&id=27777308",
            "http://www.espn.com/espn/wire?section=nhl&id=27787241",
            "http://www.espn.com/espn/wire?section=nhl&id=27781749"
        ]
        self.model = None

        self.document_index = 0

    def get_cool_stuff_inc_train_corpus(self):
        train_corpus = []
        for url in self.cool_stuff_inc_urls:
            parser = CoolStuffIncArticleParser(url)
            train_corpus.append(parser.get_tagged_document(self.document_index))

            self.document_index += 1

        return train_corpus

    def get_espn_news_wire_train_corpus(self):
        train_corpus = []
        for url in self.espn_news_wire_urls:
            parser = EspnNewsWireParser(url)
            train_corpus.append(parser.get_tagged_document(self.document_index))

            self.document_index += 1

        return train_corpus

    def train_doc2vec_model(self):

        train_corpus = []

        train_corpus = train_corpus + self.get_cool_stuff_inc_train_corpus()
        train_corpus = train_corpus + self.get_espn_news_wire_train_corpus()

        # because of the relatively small number of training examples, the number of epochs is relatively high
        # the vocab is a bunch of information about words in the document
        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
        model.build_vocab(train_corpus)

        model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

        self.model = model

    def save_doc2vec_model(self, path):
        self.model.save("new_mtg_model.model")

    def save_train_corpus(self, path):

        file = open(path + "/mtg_train_corpus.txt", "w")

        for doc in self.get_cool_stuff_inc_train_corpus():
            file.write(' '.join(doc.words))
            file.write('\n')

        file.close()

