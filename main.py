import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from algorithm import *
from path import Path, DLGConf
from pyglet.graphics import vertex_list


class SimulationWindow(pyglet.window.Window):
    def __init__(self, dlgconf: DLGConf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulation
        self.dlgconf = dlgconf
        self.path = Path(radius=150, n_points=50)

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
        # Avoid the simulation exploding at slow frames by limiting dt
        self.dlgconf.dt = min(dt, 1 / 100)
        if self.playing:
            self.path.update(self.dlgconf)
            self._update_vertex_list()
            self._update_debug_label()

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
            self.path = Path(radius=100, n_points=100)
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
            self.dlgconf.growth = min(self.dlgconf.growth + 0.001, 1.0)
        if symbol == key.W:
            self.dlgconf.growth = max(self.dlgconf.growth - 0.001, 0.0)
        if symbol == key.S:
            self.dlgconf.repulsion = min(self.dlgconf.repulsion + 1.0, 100.0)
        if symbol == key.X:
            self.dlgconf.repulsion = max(self.dlgconf.repulsion - 1.0, 1.0)

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
            color = (1, 1, 1, max(0.001, 0.05)) if self.tracing else (1, 1, 1, 1)
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
                              repulsion=40.0,
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
