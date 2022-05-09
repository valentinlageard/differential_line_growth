import numpy as np
from sklearn.neighbors import NearestNeighbors


def generate_circle(radius=1, n_points=100):
    linspace = np.linspace(-np.pi, np.pi, n_points + 1)
    circle = np.stack((np.sin(linspace) * radius, np.cos(linspace) * radius), 1)[:-1]
    return circle


def attract_to_connected(points):
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    attract_forces = (previous_points - points) + (next_points - points)
    return attract_forces


def repulse_from_neighbours(points, n_neighbours=20):
    nearest_neighbors_learner = NearestNeighbors(n_neighbors=n_neighbours)
    nearest_neighbors_learner.fit(points)
    distances, neighbours_idxs = nearest_neighbors_learner.kneighbors(n_neighbors=n_neighbours)
    all_neigbours = points[neighbours_idxs]
    broadcasted_path = np.broadcast_to(np.expand_dims(points, axis=1), all_neigbours.shape)
    vectors = broadcasted_path - all_neigbours
    distances_squared = distances ** 2
    distances_squared[distances_squared < 1.0] = 1.0  # Needed to limit jumping point glitch when points are too close
    vectors = vectors / np.broadcast_to(np.expand_dims(distances_squared, axis=2), vectors.shape)
    repulse_forces = np.sum(vectors, axis=1)
    return repulse_forces


def align(points):
    midpoints = (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) / 2.0
    align_forces = (midpoints - points)
    return align_forces


def brownian_perturbate(points):
    """
    :param points: An array of points.
    :return: An array of random forces.
    """
    return np.random.randn(*points.shape)


def sin_distribution(path, phases=1.0, offset=0.0):
    """
    :param path: A path object.
    :param phases: The number of phases of sin.
    :param offset: The offset of sin.
    :return: An array containing a sinusoidal distribution.
    """
    space = np.linspace(0 + offset, phases * np.pi * 2.0 + offset, len(path.points), endpoint=False)
    return (np.sin(space) + 1.0) / 2.0


def curve_distribution(path, offset=5, factor=2):
    """
    :param path: A path object.
    :param offset: The number of edges around a point used to compute angles.
    :return: An array containing a distribution based on curvature.
    """
    edges = np.roll(path.points, -1, axis=0) - path.points
    angles = np.sum(edges * np.roll(edges, -offset * 2 + 1, axis=0), axis=1)
    scaled_angles = np.interp(angles, (angles.min(), angles.max()), (0.0, 1.0))
    return np.roll(1.0 - scaled_angles, offset + 1, axis=0) ** factor
