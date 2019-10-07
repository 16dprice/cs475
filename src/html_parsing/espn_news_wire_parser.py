from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import lxml
import gensim


class EspnNewsWireParser:

    def __init__(self, url):
        self.url = url

    def get_clean_text(self):
        # read the webpage via a request
        # User-Agent is needed so we don't get a 403 error
        req = Request(self.url, headers={'User-Agent': 'Mozilla/5.0'})
        webpage = urlopen(req).read()

        # parse the page with lxml because other parsers don't seem to work as well
        page_soup = BeautifulSoup(webpage, "lxml")

        # select specifically by an element and class name because that's where article
        # information is stored on the cool stuff inc pages
        content = page_soup.find_all("div", attrs={"style":"overflow:hidden;"})

        clean_text = ' '.join(BeautifulSoup(content[0].encode_contents(), "lxml").stripped_strings)

        return clean_text

    def get_tagged_document(self, index=None):
        if index is None:
            return gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(self.get_clean_text()), [0])
        return gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(self.get_clean_text()), [index])
