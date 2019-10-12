import numpy as np
import matplotlib.pyplot as plt

la = np.linalg

# sentences: I like deep learning. I like NLP. I enjoy flying.
words = ["I", "like", "enjoy", "deep", "learning", "NLP", "flying", "."]

# co-occurence matrix in order of the words given above
X = np.array([[0, 2, 1, 0, 0, 0, 0, 0], # I
              [2, 0, 0, 1, 0, 1, 0, 0], # like
              [1, 0, 0, 0, 0, 0, 1, 0], # enjoy
              [0, 1, 0, 0, 1, 0, 0, 0], # deep
              [0, 0, 0, 1, 0, 0, 0, 1], # learning
              [0, 1, 0, 0, 0, 0, 0, 1], # NLP
              [0, 0, 1, 0, 0, 0, 0, 1], # flying
              [0, 0, 0, 0, 1, 1, 1, 0]])# .

# singular value decomposition
U, s, Vh = la.svd(X, full_matrices=False)

print(U)

# look at the two dimensional representation of the words
plt.axis([-0.8, 0.2, -0.8, 0.8])
for i in range(len(words)):
    plt.text(U[i, 0], U[i, 1], words[i])

plt.show()
