import random

import numpy as np
from dataclasses import dataclass, fields

from scipy.spatial import minkowski_distance
from sklearnex.neighbors import NearestNeighbors


@dataclass
class DLGConf:
    growth: float = 0.0
    attraction: float = 0.0
    repulsion: float = 0.0
    alignement: float = 0.0
    perturbation: float = 10.0
    min_distance: float = 0.0
    max_distance: float = 0.0
    scale: float = 0.0
    n_neighbours: int = 8
    growth_mode: str = 'curve'  # Can also be random or sin
    growth_mode_sin_phases: float = 3.0
    dt: float = 0.0

    def items(self):
        return iter((field.name, getattr(self, field.name)) for field in fields(self))

    def get_multiline_str(self):
        format_str = "{}: {}"
        return "\n".join(format_str.format(param, value) for param, value in self.items())


class Line:
    def __init__(self, points, is_open=False):
        self.points = points
        self.growth = np.full(len(points), 0.0)
        self.is_open = is_open

    @classmethod
    def from_circle(cls, radius=100, n_points=100, center=(0, 0), is_open=False):
        return cls(generate_circle(radius, n_points, center), is_open)


class DLGSimulation:
    def __init__(self):
        self.lines = [Line.from_circle()]
        #self.lines = [Line.from_circle(i * 10, i * 10, (random.randint(-500, 500), random.randint(-400, 400))) for i in range(2, 11)]
        self.conf = DLGConf(growth=0.005,
                              attraction=5.0,
                              repulsion=10.0,
                              alignement=5.0,
                              perturbation=0.0,
                              min_distance=2.0,
                              max_distance=50.0,
                              scale=1.0)
        self.size = sum(len(line.points) for line in self.lines)
        self.all_points = np.concatenate(list(line.points for line in self.lines))


    def reset(self):
        self.lines = [Line.from_circle()]


    def update(self, dt):
        self.conf.dt = min(dt, 1 / 100)
        self._grow()
        self._apply_forces()
        self._antialias()
        if not self.lines:
            self.reset()
        self._update_growth_distribution()
        self.all_points = np.concatenate(list(line.points for line in self.lines))
        self.size = sum(len(line.points) for line in self.lines)

    def _grow(self):
        for line in self.lines:
            random_variables = np.random.random(len(line.points))
            insertion_indexes = np.where(random_variables < line.growth, True, False).nonzero()[0]
            points = line.points[insertion_indexes]
            next_points = np.roll(line.points, 1, axis=0)[insertion_indexes]  # Here we need to adapt for open lines
            midpoints = (points + next_points) / 2.0
            line.points = np.insert(line.points, insertion_indexes, midpoints, axis=0)

    def _apply_forces(self):
        all_points = np.concatenate(list(line.points for line in self.lines))
        nearest_neighbors_learner = NearestNeighbors(n_neighbors=self.conf.n_neighbours)
        nearest_neighbors_learner.fit(all_points)
        for line in self.lines:
            align_forces = self._compute_align_force(line.points) * self.conf.alignement * self.conf.dt
            attract_forces = self._compute_attraction_force(line.points) * self.conf.attraction * self.conf.dt
            repulse_forces = self._compute_repulsion_force(line.points, all_points, nearest_neighbors_learner)
            repulse_forces *= self.conf.repulsion * self.conf.scale * self.conf.dt
            brownian_forces = self._compute_brownian_force(line.points)
            brownian_forces *= self.conf.perturbation * self.conf.scale * self.conf.dt
            all_forces = attract_forces + repulse_forces + brownian_forces + align_forces
            line.points += all_forces

    @staticmethod
    def _compute_align_force(points):
        # This wont work either for open lines
        # In align we need previous and next points
        midpoints = (np.roll(points, 1, axis=0) + np.roll(points, -1, axis=0)) / 2.0
        align_forces = (midpoints - points)
        return align_forces

    @staticmethod
    def _compute_attraction_force(points):
        # This wont work either ... !
        # In attract, we also need previous and next points
        previous_points = np.roll(points, 1, axis=0)
        next_points = np.roll(points, -1, axis=0)
        attract_forces = (previous_points - points) + (next_points - points)
        return attract_forces

    def _compute_repulsion_force(self, points, all_points, nearest_neighbors_learner):
        distances, neighbours_idxs = nearest_neighbors_learner.kneighbors(points, n_neighbors=self.conf.n_neighbours)
        all_neigbours = all_points[neighbours_idxs]
        broadcasted_path = np.broadcast_to(np.expand_dims(points, axis=1), all_neigbours.shape)
        vectors = broadcasted_path - all_neigbours
        distances_squared = np.power(distances, 2)
        # Needed to limit jumping point glitch when points are too close
        distances_squared[distances_squared < 1.0] = 1.0
        vectors = vectors / np.broadcast_to(np.expand_dims(distances_squared, axis=2), vectors.shape)
        repulse_forces = np.sum(vectors, axis=1)
        return repulse_forces

    @staticmethod
    def _compute_brownian_force(points):
        return np.random.randn(*points.shape)

    def _antialias(self):
        for line in self.lines:
            line.points = self._merge_close_points(line.points)
            line.points = self._split_long_edges(line.points)
        self.lines = [line for line in self.lines if len(line.points) > self.conf.n_neighbours]

    def _merge_close_points(self, points):
        # Here we need to manage open lines !
        distances = minkowski_distance(points, np.roll(points, -1, axis=0), p=2)
        min_distance = self.conf.min_distance * self.conf.scale  # * 2 ?
        removed_idxs = np.where(distances + np.roll(distances, 1) < min_distance)
        merged_points = np.delete(points, removed_idxs, axis=0)
        return merged_points


    def _split_long_edges(self, points):
        # TODO : Make this function better
        distances = minkowski_distance(points, np.roll(points, -1, axis=0), p=2)
        max_distance = self.conf.max_distance * self.conf.scale
        splitted_idxs = np.where(distances > max_distance)
        if len(splitted_idxs) > 1:
            left_points = points[splitted_idxs]
            right_points = np.roll(points, -1, axis=0)[splitted_idxs + 1]
            midpoints = (left_points + right_points) / 2.0
            splitted_points = np.insert(points, splitted_idxs, midpoints, axis=0)
            return splitted_points
        else:
            return points


    def _update_growth_distribution(self):
        # TODO: This is ugly
        if self.conf.growth_mode == 'curve':
            for line in self.lines:
                line.growth = curve_distribution(line.points) * self.conf.growth
        elif self.conf.growth_mode == 'sin':
            for line in self.lines:
                line.growth = sin_distribution(line.points, phases=self.conf.growth_mode_sin_phases)
                line.growth *= self.conf.growth
        else:
            for line in self.lines:
                line.growth = np.random.random(len(line.points)) * self.conf.growth


def generate_circle(radius=1, n_points=100, center=(0.0, 0.0)):
    linspace = np.linspace(-np.pi, np.pi, n_points + 1)
    circle = np.stack((np.sin(linspace) * radius + center[0], np.cos(linspace) * radius + center[1]), 1)[:-1]
    return circle


def sin_distribution(points, phases=1.0, offset=0.0):
    space = np.linspace(0 + offset, phases * np.pi * 2.0 + offset, len(points), endpoint=False)
    return (np.sin(space) + 1.0) / 2.0


def curve_distribution(points, offset=5, factor=2):
    edges = np.roll(points, -1, axis=0) - points
    angles = np.sum(edges * np.roll(edges, -offset * 2 + 1, axis=0), axis=1)
    scaled_angles = np.interp(angles, (angles.min(), angles.max()), (0.0, 1.0))
    return np.roll(1.0 - scaled_angles, offset + 1, axis=0) ** factor
