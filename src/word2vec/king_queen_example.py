import gensim.downloader as api

word_vectors = api.load("glove-wiki-gigaword-100")

file_king = open("king.txt", "w")
file_queen = open("queen.txt", "w")
file_man = open("man.txt", "w")
file_woman = open("woman.txt", "w")

for num in word_vectors['king']:
    file_king.write(str(num) + "\n")

for num in word_vectors['queen']:
    file_queen.write(str(num) + "\n")

for num in word_vectors['man']:
    file_man.write(str(num) + "\n")

for num in word_vectors['woman']:
    file_woman.write(str(num) + "\n")

file_king.close()
file_queen.close()
file_man.close()
file_woman.close()