import numpy as np
from algorithm import *
import pyglet.gl as gl
from scipy.spatial import minkowski_distance
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
        format_str = "{}: {:.3f}"
        return "\n".join(format_str.format(param, value) for param, value in self.items())


class Path:
    """Encapsulate the logic of a path and its attributes."""

    def __init__(self, *args, **kwargs):
        self.points = generate_circle(*args, **kwargs)
        self.growth_distribution = np.full(self.points.shape[0], 0.01)
        self.growth_mode = 'curve' # Can also be random or sin
        self.growth_mode_sin_phases = 3.0

    def update(self, conf):
        self.grow()
        # Quick fix to reinitialize the simulation if there are only 20 points or less
        if len(self.points) < 21:
            self.points = generate_circle(radius=150, n_points=21)
            self.growth_distribution = np.full(self.points.shape[0], 0.01)
        attract_forces = attract_to_connected(self.points) * conf.attraction * conf.dt
        repulse_forces = repulse_from_neighbours(self.points)
        repulse_forces *= conf.repulsion * conf.scale * conf.dt
        align_forces = align(self.points) * conf.alignement * conf.dt
        brownian_forces = brownian_perturbate(self.points) * conf.perturbation * conf.scale * conf.dt
        all_forces = attract_forces + repulse_forces + brownian_forces + align_forces
        self.points += all_forces
        self.merge_close_points(conf.min_distance * conf.scale)
        self.split_long_edges(conf.max_distance * conf.scale)
        if self.growth_mode == 'curve':
            self.growth_distribution = curve_distribution(self) * conf.growth
        elif self.growth_mode == 'random':
            self.growth_distribution = np.random.random(self.points.shape[0]) * conf.growth
        else:
            self.growth_distribution = sin_distribution(self, phases=self.growth_mode_sin_phases) * conf.growth

    def grow(self):
        random_variables = np.random.random(len(self.points))
        insertion_indexes = np.where(random_variables < self.growth_distribution, True, False).nonzero()[0]
        points = self.points[insertion_indexes]
        next_points = np.roll(self.points, 1, axis=0)[insertion_indexes]
        midpoints = (points + next_points) / 2.0
        self.points = np.insert(self.points, insertion_indexes, midpoints, axis=0)

    def merge_close_points(self, min_distance=1.0):
        distances = minkowski_distance(self.points, np.roll(self.points, -1, axis=0), p=2)
        removed_idxs = np.where(distances + np.roll(distances, 1) < min_distance * 2)
        merged_points = np.delete(self.points, removed_idxs, axis=0)
        self.points = merged_points

    def split_long_edges(self, max_distance=10.0):
        # TODO : Make this function better
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
