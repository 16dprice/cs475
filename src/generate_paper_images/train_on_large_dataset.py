import time
import multiprocessing
from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.word2vec import Word2Vec


class WikiSentences:
    def __init__(self, wiki_dump_path):
        self.wiki = WikiCorpus(wiki_dump_path)

    def __iter__(self):
        for sentence in self.wiki.get_texts():
            yield sentence


start = time.time()
print('getting sentences...')
wiki_sentences = WikiSentences('data/en/enwiki-latest-pages-articles.xml.bz2')
print('got sentences in {} seconds'.format(time.time() - start))

params = {
    'size': 200,
    'window': 10,
    'min_count': 10,
    'workers': max(1, multiprocessing.cpu_count() - 1),
    'sample': 1E-3
}

start = time.time()
print('training model...')

model = Word2Vec(wiki_sentences, **params)

end = time.time()
print('trained model in {} seconds'.format(end - start))

model.save('wiki-dump-model.model')
print('saved model in {} seconds'.format(time.time() - end))