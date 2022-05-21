import configparser

import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from pyglet.graphics import vertex_list
from scipy.interpolate import interp1d
from colour import Color
import numpy as np
import click
import mido

from simulation import DLGSimulation


def select_input_port_name():
    input_port_names = mido.get_input_names()
    print('Input ports :')
    for i, input_port_name in enumerate(input_port_names):
        print(' ', i, input_port_name)
    prompt = 'Select a port name [0-{}]'.format(len(input_port_names) - 1)
    input_port_name_index = click.prompt(prompt, type=click.IntRange(0, len(input_port_names) - 1))
    return input_port_names[input_port_name_index]


class SimulationWindow(pyglet.window.Window):
    def __init__(self, midi_conf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.simulation = DLGSimulation()

        # Color
        self.hue = 1
        self.saturation = 1
        self.luminance = 1

        # Midi
        if midi_conf['midi']['enabled'] == 'yes':
            self.midi_interaction = True
            self.inport = mido.open_input(select_input_port_name())
            self.midi_conf = midi_conf
        else:
            self.midi_interaction = False
            self.inport = None
            self.midi_conf = None

        # Rendering
        self.line_vertex_list = vertex_list(0, ('v2f', ()), ('c4f', ()))
        self.node_vertex_list = vertex_list(0, ('v2f', ()), ('c4f', ()))
        gl.glLineWidth(1)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        # Debug
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.fps_display.label.color = (255, 255, 255, 255)
        self.debug_label = pyglet.text.Label(x=5, y=self.height - 15, width=400, height=400, multiline=True)

        # Modes
        self.playing = True
        self.tracing = False
        self.debug_info = False
        self.debug_color = False
        self.node_drawing = False

        pyglet.clock.schedule_interval(self.update, 1 / 240.0)
        self.clear()

    def update(self, dt):
        if self.midi_interaction:
            self._update_midi()
        if self.playing:
            self.simulation.update(dt)
            self._update_vertices()
            if self.debug_info:
                self._update_debug_label()

    def on_draw(self):
        if not self.tracing:
            self.clear()
        if self.debug_info and not self.tracing:
            self.fps_display.draw()
            self.debug_label.draw()
        if self.node_drawing:
            self.node_vertex_list.draw(gl.GL_POINTS)
        else:
            self.line_vertex_list.draw(gl.GL_LINES)

    def on_close(self):
        exit()

    def _update_vertices(self):
        if not self.node_drawing:
            formatted_vertices = self._format_vertices()
            centered_points = (formatted_vertices + np.array([self.width / 2, self.height / 2])).flatten()
            self.line_vertex_list.resize(len(formatted_vertices))
            self.line_vertex_list.vertices = centered_points
            # TODO : debug color !
            self.line_vertex_list.colors = self._get_color() * len(formatted_vertices)
        else:
            centered_points = (self.simulation.all_points + np.array([self.width / 2, self.height / 2]))
            self.node_vertex_list.resize(len(centered_points))
            self.node_vertex_list.vertices = centered_points.flatten()
            self.node_vertex_list.colors = self._get_color() * len(centered_points)

    def _get_color(self):
        r, g, b = Color(hue=self.hue, saturation=self.saturation, luminance=self.luminance).rgb
        return (r, g, b, max(0.001, 0.05)) if self.tracing else (r, g, b, 1)

    def _format_vertices(self):
        # This is fucking ugly
        formatted_size = sum(2 * (len(line.points) - 1) if line.is_open else 2 * len(line.points)
                             for line in self.simulation.lines)
        formatted_vertices = np.empty((formatted_size, 2), dtype='float32')
        prev_idx = 0
        for line in self.simulation.lines:
            last_idx = 0
            if line.is_open:
                last_idx = (2 * (len(line.points) - 1))
                extended_points = np.repeat(line.points, 2)[1:-1]
            else:
                last_idx = 2 * len(line.points)
                extended_points = np.insert(line.points, np.arange(len(line.points)), np.roll(line.points, 1, axis=0),
                                            axis=0)
            formatted_vertices[prev_idx:prev_idx + last_idx] = extended_points
            prev_idx += last_idx
        return formatted_vertices

    def _update_debug_label(self):
        n_nodes = self.simulation.size
        self.debug_label.text = "Nodes: {}\n".format(n_nodes) + self.simulation.conf.get_multiline_str()

    def _update_midi(self):
        for msg in self.inport.iter_pending():
            if msg.type == 'control_change':
                if msg.control == int(self.midi_conf['midi']['cc_scale']):
                    self.simulation.conf.scale = 0.5 + (interp1d([0, 127], [0, 1])(msg.value) ** 0.5) * (5 - 0.5)
                if msg.control == int(self.midi_conf['midi']['cc_growth']):
                    self.simulation.conf.growth = interp1d([0, 127], [0, 0.1])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_repulsion']):
                    self.simulation.conf.repulsion = interp1d([0, 127], [1, 50])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_attraction']):
                    self.simulation.conf.attraction = interp1d([0, 127], [1, 20])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_alignement']):
                    self.simulation.conf.alignement = interp1d([0, 127], [1, 20])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_growth_sin_phases']):
                    self.simulation.conf.growth_mode_sin_phases = float(int(interp1d([0, 127], [1, 8])(msg.value)))
                if msg.control == int(self.midi_conf['midi']['cc_hue']):
                    self.hue = interp1d([0, 127], [0, 1])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_saturation']):
                    self.saturation = interp1d([0, 127], [0, 1])(msg.value)
                if msg.control == int(self.midi_conf['midi']['cc_luminance']):
                    self.luminance = interp1d([0, 127], [0.1, 1])(msg.value)
            elif msg.type == 'note_on':
                if msg.note == int(self.midi_conf['midi']['note_reset']):
                    self.clear()
                    self.simulation.reset()
                if msg.note == int(self.midi_conf['midi']['note_play_pause']):
                    self.playing = not self.playing
                if msg.note == int(self.midi_conf['midi']['note_debug_info']):
                    self.debug_info = not self.debug_info
                if msg.note == int(self.midi_conf['midi']['note_trace']):
                    self.tracing = not self.tracing
                    self.clear()
                if msg.note == int(self.midi_conf['midi']['note_debug_color']):
                    self.debug_color = not self.debug_color
                if msg.note == int(self.midi_conf['midi']['note_node_draw']):
                    self.node_drawing = not self.node_drawing
                if msg.note == int(self.midi_conf['midi']['note_curve_growth']):
                    self.simulation.conf.growth_mode = 'curve'
                if msg.note == int(self.midi_conf['midi']['note_random_growth']):
                    self.simulation.conf.growth_mode = 'random'
                if msg.note == int(self.midi_conf['midi']['note_sin_growth']):
                    self.simulation.conf.growth_mode = 'sin'

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.playing = not self.playing
        if symbol == key.A:
            self.tracing = not self.tracing
            self.clear()
        if symbol == key.R:
            self.clear()
            self.simulation.reset()
        if symbol == key.I:
            self.debug_info = not self.debug_info
        if symbol == key.O:
            self.debug_color = not self.debug_color
        if symbol == key.P:
            self.node_drawing = not self.node_drawing
        if symbol == key.UP:
            self.simulation.conf.scale = min(self.simulation.conf.scale * 1.1, 20.0)
        if symbol == key.DOWN:
            self.simulation.conf.scale = max(self.simulation.conf.scale / 1.1, 0.01)
        if symbol == key.Q:
            self.simulation.conf.growth = min(self.simulation.conf.growth + 0.001, 0.1)
        if symbol == key.W:
            self.simulation.conf.growth = max(self.simulation.conf.growth - 0.001, 0.0)
        if symbol == key.S:
            self.simulation.conf.attraction = min(self.simulation.conf.attraction + 1.0, 20.0)
        if symbol == key.X:
            self.simulation.conf.attraction = max(self.simulation.conf.attraction - 1.0, 1.0)
        if symbol == key.D:
            self.simulation.conf.repulsion = min(self.simulation.conf.repulsion + 1.0, 50.0)
        if symbol == key.C:
            self.simulation.conf.repulsion = max(self.simulation.conf.repulsion - 1.0, 1.0)
        if symbol == key.F:
            self.simulation.conf.alignement = min(self.simulation.conf.alignement + 1.0, 20.0)
        if symbol == key.V:
            self.simulation.conf.alignement = max(self.simulation.conf.alignement - 1.0, 1.0)


def main():
    midi_conf = configparser.ConfigParser()
    midi_conf.read('config.ini')
    window_conf = pyglet.gl.Config(sample_buffers=1, sample=1)
    window = SimulationWindow(midi_conf, fullscreen=True, config=window_conf)
    pyglet.app.run()


if __name__ == '__main__':
    main()
