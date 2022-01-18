import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass, fields


@dataclass
class DLGConf:
    growth: float = 0.0
    attraction: float = 0.0
    repulsion: float = 0.0
    alignement: float = 0.0
    perturbation: float = 0.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    scale: float = 0.0
    dt: float = 0.0

    def items(self):
        return iter((field.name, getattr(self, field.name)) for field in fields(self))

    def get_multiline_str(self):
        format_str = "{}: {:.2f}"
        return "\n".join(format_str.format(param, value) for param, value in self.items())


def generate_circle(radius=1, n_points=100):
    linspace = np.linspace(-np.pi, np.pi, n_points + 1)
    circle = np.stack((np.sin(linspace) * radius, np.cos(linspace) * radius), 1)[:-1]
    return circle


def attract_to_connected(points):
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    attract_forces = (previous_points - points) + (next_points - points)
    return attract_forces

def repulse_from_neighbours(points, n_neighbours=100):
    nearest_neighbors_learner = NearestNeighbors(n_neighbors=n_neighbours)
    nearest_neighbors_learner.fit(points)
    distances, neighbours_idxs = nearest_neighbors_learner.kneighbors(n_neighbors=20)
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
    return np.random.randn(*points.shape)


def sin_distribution(path, phases=1.0, offset=0.0, p=0.05):
    space = np.linspace(0 + offset, phases * np.pi * 2.0 + offset, len(path.points), endpoint=False)
    return (np.sin(space) + 1.0) / 2.0 * p
