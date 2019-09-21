from html_parsing.cool_stuff_inc_article_parser import CoolStuffIncArticleParser
import gensim
import collections
import random

urls = [
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

train_corpus = []
for index, url in enumerate(urls):
    parser = CoolStuffIncArticleParser(url)
    train_corpus.append(parser.get_tagged_document(index))

# because of the relatively small number of training examples, the number of epochs is relatively high
# the vocab is a bunch of information about words in the document
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=100)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# test_url = "https://www.coolstuffinc.com/a/jimdavis-06112019-starting-with-an-idea-modern-four-color-saheeli-combo"
# parser = CoolStuffIncArticleParser(test_url)
#
# inferred_vector = model.infer_vector(parser.get_tagged_document(20).words)
#
# sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#
# print(sims)

# assess the model
ranks = []
second_ranks = []

# this is more of a 'sanity check' rather than a real accuracy test
# this basically just checks to see if the model acts as we expect
# that is, documents should be very closely related to themselves
# if they're not, something went terribly wrong
# TODO: check which ones are populating the second_ranks array (look at which documents aren't ranking 'sanely')
for doc_id in range(len(train_corpus)):

    # infer a vector from something that we already know about
    # if the model is good, this should be extremely closely related to itself in the model
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)

    # returns a list of form [(int, float), ... ] where the int is the doc_id and
    # the float is cosine similarity
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

    # the smaller the index of the doc_id, the better the model is
    # so, a rank of 0 is very good
    # this would mean that the model predicted the document itself as the closest document
    rank = [docid for docid, sim in sims].index(doc_id)

    # if this is a list of 0's, the model has done very well
    ranks.append(rank)

    # the second closest document
    second_ranks.append(sims[1])

# an easy way to see if the model is 'sane'
# should expect a large number of 0's to be present in ranks
print(collections.Counter(ranks))

# pick a random document from the corpus and infer a vector from the model
doc_id = random.randint(0, len(train_corpus) - 1)

# sometimes this is pretty bad because a document can be 'isolated'
# that is, the model may be trained well, but there does not exist a document very related to the chosen document at all
print('Train Document ({}): <<{}>>\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
sim_id, sim_dist = second_ranks[doc_id]  # the second most similar doc
print('Similar Document ({}, {}): <<{}>>\n'.format(sim_id, sim_dist, ' '.join(train_corpus[sim_id].words)))

model.save("./new_mtg_model.model")

