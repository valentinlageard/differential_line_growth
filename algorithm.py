import itertools
import numpy as np
from scipy.spatial import KDTree, minkowski_distance

def generate_circle(radius=1, n_points=100):
    linspace = np.linspace(-np.pi, np.pi, n_points + 1)
    circle = np.stack((np.sin(linspace) * radius, np.cos(linspace) * radius), 1)[:-1]
    return circle


def attract_to_connected(path):
    previous_points = np.roll(path, 1, axis=0)
    next_points = np.roll(path, -1, axis=0)
    attract_forces = (previous_points - path) + (next_points - path)
    return attract_forces


def repulse_from_neighbours(path, radius=10.0):
    kdtree = KDTree(path, balanced_tree=False)
    neighbours_idxs = kdtree.query_ball_tree(kdtree, radius)
    neighbours_idxs = np.array(list(itertools.zip_longest(*neighbours_idxs, fillvalue=-1))).T
    path_padded = np.vstack([path, np.array([0.0, 0.0])])
    all_neigbours = path_padded[neighbours_idxs]
    broadcasted_path = np.broadcast_to(np.expand_dims(path, axis=1), all_neigbours.shape)
    all_neigbours_zero_is_self = np.where(all_neigbours != [0, 0], all_neigbours, broadcasted_path)
    vectors = broadcasted_path - all_neigbours_zero_is_self
    repulse_forces = np.sum(vectors, axis=1)
    return repulse_forces


def align_to_connected(path):
    midpoints = (np.roll(path, 1, axis=0) + np.roll(path, -1, axis=0)) / 2.0
    align_forces = (midpoints - path)
    return align_forces


def brownian_perturbate(path):
    return np.random.randn(*path.shape)


def split_merge(path, split_distance=10.0, merge_distance=1.0):
    distances = minkowski_distance(path, np.roll(path, -1, axis=0), p=2)
    n_new_points = distances[distances > split_distance].shape[0]
    n_removed_points = distances[distances < merge_distance].shape[0]
    new_path = np.zeros((path.shape[0] + n_new_points - n_removed_points, 2))
    counter = 0
    for i, (point, distance_to_next) in enumerate(zip(path, distances)):
        if distance_to_next > merge_distance:
            new_path[i + counter] = point
        else:
            counter -= 1
        if distance_to_next > split_distance:
            counter += 1
            new_point = (point + path[(i + 1) % path.shape[0]]) / 2.0
            new_path[i + counter] = new_point
    return new_path


def random_grow(path, new_points_per_frame=1):
    insertion_indexes = np.random.randint(0, path.shape[0], new_points_per_frame)
    points = path[insertion_indexes]
    next_points = np.roll(path, -1, axis=0)[insertion_indexes]
    midpoints = (points + next_points) / 2.0
    return np.insert(path, insertion_indexes, midpoints, axis=0)


def differential_growth(path, attraction_strength=0.1, repulsion_strength=0.1, repulsion_radius=10.0,
                        brownian_strength=1.0, align_strength=0.1, split_distance=10.0, merge_distance=1.0):
    attract_forces = attract_to_connected(path) * attraction_strength
    repulse_forces = repulse_from_neighbours(path, repulsion_radius) * repulsion_strength
    brownian_forces = brownian_perturbate(path) * brownian_strength
    align_forces = align_to_connected(path) * align_strength
    new_path = split_merge(path + attract_forces + repulse_forces + brownian_forces + align_forces,
                           split_distance, merge_distance)
    new_path = random_grow(new_path)
    return new_path
