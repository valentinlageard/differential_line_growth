import numpy as np
from algorithm import *
import pyglet.gl as gl


class Path:
    """Encapsulate the logic of a path and its attributes."""

    def __init__(self, *args, **kwargs):
        self.points = generate_circle(*args, **kwargs)
        self.growth_distribution = np.full(self.points.shape[0], 0.01)
        # self.lifetimes = None

    def update(self, conf, supersampling=1):
        conf.dt /= supersampling
        for i in range(supersampling):
            self.grow()
            attract_forces = attract_to_connected(self.points) * conf.attraction * conf.scale * conf.dt
            repulse_forces = repulse_from_neighbours(self.points, conf.repulsion * conf.scale * conf.dt,
                                                 conf.repulsion_radius * conf.scale)
            align_forces = align(self.points) * conf.alignement * conf.scale * conf.dt
            brownian_forces = brownian_perturbate(self.points) * conf.perturbation * conf.scale * conf.dt
            all_forces = attract_forces + repulse_forces + brownian_forces + align_forces
            self.points += all_forces
            self.merge_close_points(conf.min_distance * conf.scale)
            self.split_long_edges(conf.max_distance * conf.scale)
            #self.growth_distribution = sin_distribution(self, phases=3.0, p=conf.growth)
            self.growth_distribution = np.random.random(self.points.shape[0]) * conf.growth

    def grow(self):
        random_variables = np.random.random(len(self.points))
        insertion_indexes = np.where(random_variables < self.growth_distribution, True, False).nonzero()[0]
        points = self.points[insertion_indexes]
        next_points = np.roll(self.points, -1, axis=0)[insertion_indexes]
        midpoints = (points + next_points) / 2.0
        self.points = np.insert(self.points, insertion_indexes, midpoints, axis=0)

    def merge_close_points(self, min_distance=1.0):
        distances = minkowski_distance(self.points, np.roll(self.points, -1, axis=0), p=2)
        # n_removed_points = distances[distances < min_distance and np.roll(distances, -1) < min_distance]
        removed_idxs = np.where(distances + np.roll(distances, 1) < min_distance * 2)
        merged_points = np.delete(self.points, removed_idxs, axis=0)
        self.points = merged_points

    def split_long_edges(self, max_distance=10.0):
        distances = minkowski_distance(self.points, np.roll(self.points, -1, axis=0), p=2)
        splitted_idxs = np.where(distances > max_distance)
        if len(splitted_idxs) > 1:
            left_points = self.points[splitted_idxs]
            right_points = np.roll(self.points, -1, axis=0)[splitted_idxs + 1]
            midpoints = (left_points + right_points) / 2.0
            splitted_points = np.insert(self.points, splitted_idxs, midpoints, axis=0)
            self.points = splitted_points

    def get_centered_points(self, width, height):
        return self.points + np.array([width / 2, height / 2])

    def __len__(self):
        return self.points.shape[0]
