import gensim

model = gensim.models.doc2vec.Doc2Vec.load("new_model.d2v")
print model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])

