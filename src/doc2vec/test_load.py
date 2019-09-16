import gensim

model = gensim.models.doc2vec.Doc2Vec.load("doc2vec/new_model.model")
inferred_vector = model.infer_vector(['only', 'you', 'can', 'prevent', 'forest', 'fires'])

print(model.docvecs.most_similar([inferred_vector], topn=3))
