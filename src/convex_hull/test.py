import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

np.random.seed(0)

points = np.random.rand(30, 2)
hull = ConvexHull(points)


plt.plot(points[:,0], points[:,1], 'o')

for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

point = np.array([
    0.01,
    0.15
])

plt.plot(point[0], point[1], 'o')

hull = Delaunay(points)
print(hull.find_simplex(point) >= 0)

plt.show()