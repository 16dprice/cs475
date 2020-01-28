import numpy as np
import gensim
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
#
# class ConvexHull:
#
#     def __init__(self, hull):
#         self.hull = hull
#
#     @staticmethod
#     def load_by_model(model):
#         doc_vectors = np.array(model.docvecs.vectors_docs)
#         return ConvexHull(doc_vectors)


model = gensim.models.doc2vec.Doc2Vec.load("../doc2vec_models/nov10_19/vs_25_epochs_100/aggregate_model.model")
# hull_obj = ConvexHull.load_by_model(model)

monte_carlo_sample_size = 2

doc_vectors = np.array(model.docvecs.vectors_docs)

dims_mins_maxs = []
for i in range(25):
    min_val = min(doc_vectors[0:, i])
    max_val = max(doc_vectors[0:, i])

    dims_mins_maxs.append([min_val, max_val])

# generate random points in n-dimensional space based on min and max values determined in previous step
rand_points = np.zeros((monte_carlo_sample_size, 25))
for i in range(monte_carlo_sample_size):
    for j in range(25):
        rand_points[i, j] = np.random.uniform(np.floor(dims_mins_maxs[j][0]), np.ceil(dims_mins_maxs[j][1]))

magic_points = doc_vectors[:50]
sports_points = doc_vectors[50:88]
dance_points = doc_vectors[88:]

points_in_magic = 0
points_in_sports = 0
points_in_dance = 0

points_in_magic_and_sports = 0
points_in_sports_and_dance = 0
points_in_dance_and_magic = 0

points_in_all = 0

print("here0")
magic_hull = Delaunay(magic_points)
sports_hull = Delaunay(sports_points)
dance_hull = Delaunay(dance_points)
for i in range(monte_carlo_sample_size):
    print("here1")
    in_magic = magic_hull.find_simplex(np.array(rand_points[i])) >= 0
    in_sports = sports_hull.find_simplex(np.array(rand_points[i])) >= 0
    in_dance = dance_hull.find_simplex(np.array(rand_points[i])) >= 0
    print("here2")

    if in_magic: points_in_magic += 1
    if in_sports: points_in_sports += 1
    if in_dance: points_in_dance += 1

    if in_magic and in_sports: points_in_magic_and_sports += 1
    if in_sports and in_dance: points_in_sports_and_dance += 1
    if in_dance and in_magic: points_in_dance_and_magic += 1

    if in_magic and in_sports and in_dance: points_in_all += 1

points_in_any = points_in_magic + points_in_sports + points_in_dance

hyper_cube_area = 1
for i in range(25):
    hyper_cube_area *= abs(dims_mins_maxs[i][0] - dims_mins_maxs[i][1])

total_area = hyper_cube_area * (points_in_any / monte_carlo_sample_size)

magic_area = hyper_cube_area * (points_in_magic / monte_carlo_sample_size)
sports_area = hyper_cube_area * (points_in_sports / monte_carlo_sample_size)
dance_area = hyper_cube_area * (points_in_dance / monte_carlo_sample_size)

magic_sports_area = hyper_cube_area * (points_in_magic_and_sports / monte_carlo_sample_size)
sports_dance_area = hyper_cube_area * (points_in_sports_and_dance / monte_carlo_sample_size)
dance_magic_area = hyper_cube_area * (points_in_dance_and_magic / monte_carlo_sample_size)

magic_sports_dance_area = hyper_cube_area * (points_in_all / monte_carlo_sample_size)

description = "Hyper Cube Area: {}\n" \
              "Total Convex Area: {}\n\n" \
              "Total M Area: {}\n" \
              "Total S Area: {}\n" \
              "Total B Area: {}\n\n" \
              "Total MS Shared Area: {}\n" \
              "Total SB Shared Area: {}\n" \
              "Total BM Shared Area: {}\n\n" \
              "Total All Shared Area: {}\n".format(
    round(hyper_cube_area),
    round(total_area, 2),
    round(magic_area, 2),
    round(sports_area, 2),
    round(dance_area, 2),
    round(magic_sports_area, 2),
    round(sports_dance_area, 2),
    round(dance_magic_area, 2),
    round(magic_sports_dance_area, 2)
)

print(description)