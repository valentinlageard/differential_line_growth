import itertools
import numpy as np
from scipy.spatial import KDTree, minkowski_distance
from dataclasses import dataclass


@dataclass
class DLGConf:
    growth: float = 0.0
    attraction: float = 0.0
    repulsion: float = 0.0
    alignement: float = 0.0
    perturbation: float = 0.0
    repulsion_radius: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    scale: float = 0.0
    dt: float = 0.0


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
    # Variable list of list to numpy matrix
    neighbours_idxs = np.array(list(itertools.zip_longest(*neighbours_idxs, fillvalue=-1))).T
    # Pad the last row with a default value so fillvalues index there
    path_padded = np.vstack([path, np.array([0.0, 0.0])])
    all_neigbours = path_padded[neighbours_idxs]
    # Broadcast for substraction
    broadcasted_path = np.broadcast_to(np.expand_dims(path, axis=1), all_neigbours.shape)
    # Replace default values by the point so substraction gives [0, 0]
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


def constrain_distance(path, min_distance=1.0, max_distance=10.0):
    distances = minkowski_distance(path, np.roll(path, -1, axis=0), p=2)
    n_new_points = distances[distances > max_distance].shape[0]
    n_removed_points = distances[distances < min_distance].shape[0]
    new_path = np.zeros((path.shape[0] + n_new_points - n_removed_points, 2))
    counter = 0
    for i, (point, distance_to_next) in enumerate(zip(path, distances)):
        if distance_to_next > min_distance:
            new_path[i + counter] = point
        else:
            counter -= 1
        if distance_to_next > max_distance:
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


def sin_grow(path, phases=1.0, offset=0.0, max_p=0.05):
    space = np.linspace(0 + offset, phases * np.pi * 2.0 + offset, path.shape[0], endpoint=False)
    sin_distribution = (np.sin(space) + 1.0) / 2.0 * max_p
    random_variables = np.random.random(path.shape[0])
    insertion_indexes = np.where(random_variables < sin_distribution, True, False).nonzero()[0]
    points = path[insertion_indexes]
    next_points = np.roll(path, -1, axis=0)[insertion_indexes]
    midpoints = (points + next_points) / 2.0
    return np.insert(path, insertion_indexes, midpoints, axis=0)


def differential_line_growth(path, conf: DLGConf):
    grown_path = sin_grow(path, phases=3.0, max_p=conf.growth)
    attract_forces = attract_to_connected(grown_path) * conf.attraction * conf.scale * conf.dt
    repulse_forces = repulse_from_neighbours(grown_path, conf.repulsion_radius * conf.scale)\
                     * conf.repulsion * conf.scale * conf.dt
    align_forces = align_to_connected(grown_path) * conf.alignement * conf.scale * conf.dt
    brownian_forces = brownian_perturbate(grown_path) * conf.perturbation * conf.scale * conf.dt
    path_with_forces = grown_path + attract_forces + repulse_forces + brownian_forces + align_forces
    new_path = constrain_distance(path_with_forces, conf.min_distance * conf.scale, conf.max_distance * conf.scale)
    return new_path
