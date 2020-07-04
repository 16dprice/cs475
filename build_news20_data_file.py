import gensim
import os

file = open("/home/dj/PycharmProjects/cs475/src/data/20news_corpus.txt", "w")

def get_20news_docs(path):

    doc_index = 0

    for file_name in os.listdir(path):
        try:
            doc_index += 1

            data = open(path + "/" + file_name, "r")
            text = ""

            for line in data:
                text += line

            tagged_doc = gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(text), [doc_index])
            file.write(' '.join(tagged_doc.words))
            file.write('\n')

        except:
            print(file_name)

    print(doc_index)

get_20news_docs("/home/dj/PycharmProjects/cs475/src/data/20news-bydate-train/soc.religion.christian") # 599
get_20news_docs("/home/dj/PycharmProjects/cs475/src/data/20news-bydate-train/misc.forsale") # 585