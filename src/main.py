from html_parsing.html_parser import CoolStuffIncArticleParser
import gensim

urls = [
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
    "https://www.coolstuffinc.com/a/jimdavis-07122019-combos-to-watch-for-in-week-one-of-core-set-2020-standard"
]

train_corpus = []
for index, url in enumerate(urls):
    parser = CoolStuffIncArticleParser(url)
    train_corpus.append(parser.get_tagged_document(index))

# because of the relatively small number of training examples, the number of epochs is relatively high
# the vocab is a bunch of information about words in the document
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

test_url = "https://www.coolstuffinc.com/a/jimdavis-06112019-starting-with-an-idea-modern-four-color-saheeli-combo"
parser = CoolStuffIncArticleParser(test_url)

inferred_vector = model.infer_vector(parser.get_tagged_document(20).words)

sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

print(sims)
