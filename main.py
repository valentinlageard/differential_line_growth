import mido
import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from pyglet.graphics import vertex_list
from scipy.interpolate import interp1d
from colour import Color

from algorithm import *
from path import Path, DLGConf
import click
import mido

def select_input_port_name():
    input_port_names = mido.get_input_names()
    print('Input ports :')
    for i, input_port_name in enumerate(input_port_names):
        print(' ', i, input_port_name)
    prompt = 'Select a port name [0-{}]'.format(len(input_port_names) - 1)
    input_port_name_index = click.prompt(prompt, type=click.IntRange(0, len(input_port_names) - 1))
    return input_port_names[input_port_name_index]

class SimulationWindow(pyglet.window.Window):
    def __init__(self, dlgconf: DLGConf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulation
        self.dlgconf = dlgconf
        self.path = Path(radius=150, n_points=50)

        # Color
        self.color = Color("white")
        self.hue = 1
        self.saturation = 1
        self.luminance = 1

        # Midi
        self.inport = mido.open_input(select_input_port_name())

        # Rendering
        centered_path = self.path.get_centered_points(self.width, self.height)
        self.vertex_list = vertex_list(len(self.path),
                                       ('v2f', list(centered_path.flatten())),
                                       ('c4f', (1, 1, 1, 0.01) * len(self.path)))
        self.vertex_list_points = vertex_list(len(self.path),
                                              ('v2f', list(centered_path.flatten())),
                                              ('c4f', (0, 1, 0, 1) * len(self.path)))

        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.fps_display.label.color = (255, 255, 255, 255)
        self.debug_label = pyglet.text.Label(x=5, y=self.height - 15, width=400, height=400, multiline=True)

        gl.glLineWidth(1)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.playing = True
        self.tracing = False
        self.debug_info = False
        self.debug_color = False
        self.node_drawing = False

        pyglet.clock.schedule_interval(self.update, 1 / 240.0)
        self.clear()

    def update(self, dt):
        self.update_midi()
        # Avoid the simulation exploding at slow frames by limiting dt
        self.dlgconf.dt = min(dt, 1 / 100)
        if self.playing:
            self.path.update(self.dlgconf)
            self._update_vertex_list()
            self._update_debug_label()

    def update_midi(self):
        for msg in self.inport.iter_pending():
            if msg.type == 'control_change':
                if msg.control == 19:
                    self.dlgconf.scale = 0.5 + (interp1d([0, 127], [0, 1])(msg.value) ** 0.5) * (5 - 0.5)
                if msg.control == 23:
                    self.dlgconf.growth = interp1d([0, 127], [0, 0.5])(msg.value)
                if msg.control == 27:
                    self.dlgconf.repulsion = interp1d([0, 127], [1, 50])(msg.value)
                if msg.control == 31:
                    self.dlgconf.attraction = interp1d([0, 127], [1, 20])(msg.value)
                if msg.control == 49:
                    self.dlgconf.alignement = interp1d([0, 127], [1, 20])(msg.value)
                if msg.control == 53:
                    self.path.growth_mode_sin_phases = float(int(interp1d([0, 127], [1, 8])(msg.value)))
                if msg.control == 57:
                    self.hue = interp1d([0, 127], [0, 1])(msg.value)
                if msg.control == 61:
                    self.saturation = interp1d([0, 127], [0, 1])(msg.value)
                if msg.control == 62:
                    self.luminance = interp1d([0, 127], [0.1, 1])(msg.value)
            elif msg.type == 'note_on':
                if msg.note == 1:
                    self.clear()
                    self.path = Path(radius=100, n_points=100)
                if msg.note == 3:
                    self.playing = not self.playing
                if msg.note == 6:
                    self.tracing = not self.tracing
                    self.clear()
                if msg.note == 4:
                    self.debug_info = not self.debug_info
                if msg.note == 7:
                    self.debug_color = not self.debug_color
                if msg.note == 9:
                    self.node_drawing = not self.node_drawing
                if msg.note == 10:
                    self.path.growth_mode = 'curve'
                if msg.note == 12:
                    self.path.growth_mode = 'random'
                if msg.note == 13:
                    self.path.growth_mode = 'sin'

    def on_draw(self):
        if not self.tracing:
            self.clear()
        if self.debug_info and not self.tracing:
            self.fps_display.draw()
            self.debug_label.draw()
        if self.node_drawing:
            self.vertex_list.draw(gl.GL_POINTS)
        else:
            self.vertex_list.draw(gl.GL_LINE_LOOP)

    def on_close(self):
        exit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.playing = not self.playing
        if symbol == key.A:
            self.tracing = not self.tracing
            self.clear()
        if symbol == key.R:
            self.clear()
            self.path.points = generate_circle(radius=150, n_points=21)
            #self.path.growth_distribution = np.full(self.points.shape[0], 0.01)
        if symbol == key.I:
            self.debug_info = not self.debug_info
        if symbol == key.O:
            self.debug_color = not self.debug_color
        if symbol == key.P:
            self.node_drawing = not self.node_drawing
        if symbol == key.UP:
            self.dlgconf.scale = min(self.dlgconf.scale * 1.1, 20.0)
        if symbol == key.DOWN:
            self.dlgconf.scale = max(self.dlgconf.scale / 1.1, 0.01)
        if symbol == key.Q:
            self.dlgconf.growth = min(self.dlgconf.growth + 0.001, 0.05)
        if symbol == key.W:
            self.dlgconf.growth = max(self.dlgconf.growth - 0.001, 0.0)
        if symbol == key.S:
            self.dlgconf.attraction = min(self.dlgconf.attraction + 1.0, 50.0)
        if symbol == key.X:
            self.dlgconf.attraction = max(self.dlgconf.attraction - 1.0, 1.0)
        if symbol == key.D:
            self.dlgconf.repulsion = min(self.dlgconf.repulsion + 1.0, 50.0)
        if symbol == key.C:
            self.dlgconf.repulsion = max(self.dlgconf.repulsion - 1.0, 1.0)
        if symbol == key.F:
            self.dlgconf.alignement = min(self.dlgconf.alignement + 1.0, 50.0)
        if symbol == key.V:
            self.dlgconf.alignement = max(self.dlgconf.alignement - 1.0, 1.0)

    def _update_vertex_list(self):
        centered_path = self.path.get_centered_points(self.width, self.height)
        self.vertex_list.resize(len(self.path))
        self.vertex_list.vertices = list(centered_path.flatten())
        if self.debug_color:
            ones = np.full(len(self.path), 1.0)
            inverscaled_distribution = 1 - (self.path.growth_distribution / self.dlgconf.growth)
            alphas = ones / 100 * self.dlgconf.scale if self.tracing else ones
            colors = np.ravel(np.column_stack((ones, inverscaled_distribution, inverscaled_distribution, alphas)))
            self.vertex_list.colors = colors
        else:
            self.color = Color(hue=self.hue, saturation=self.saturation, luminance=self.luminance)
            r, g, b = self.color.rgb
            print(r, g, b)
            color = (r, g, b, max(0.001, 0.05)) if self.tracing else (r, g, b, 1)
            self.vertex_list.colors = color * len(self.path)
        if self.node_drawing:
            self.vertex_list_points.resize(len(self.path))
            self.vertex_list_points.vertices = list(centered_path.flatten())
            self.vertex_list_points.colors = (0, 1, 0, 1) * len(self.path)

    def _update_debug_label(self):
        self.debug_label.text = "Nodes: {}\n".format(len(self.path)) + self.dlgconf.get_multiline_str()


def main():
    simulation_conf = DLGConf(growth=0.005,
                              attraction=5.0,
                              repulsion=10.0,
                              alignement=5.0,
                              perturbation=5.0,
                              min_distance=1.0,
                              max_distance=50.0,
                              scale=1.0)
    window_conf = pyglet.gl.Config(sample_buffers=1, sample=4)
    window = SimulationWindow(dlgconf=simulation_conf, fullscreen=True, config=window_conf)
    pyglet.app.run()


if __name__ == '__main__':
    main()
