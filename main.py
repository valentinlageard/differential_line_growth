import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from algorithm import *
from path import Path


class SimulationWindow(pyglet.window.Window):
    def __init__(self, dlgconf: DLGConf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dlgconf = dlgconf
        self.path = Path(radius=100, n_points=100)

        self.batch = pyglet.graphics.Batch()
        centered_path = self.path.get_centered_points(self.width, self.height)
        self.vertex_list = self.batch.add(len(self.path), gl.GL_LINE_LOOP, None,
                                          ('v2f', list(centered_path.flatten())),
                                          ('c4f', (1, 1, 1, 0.01) * len(self.path)))

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
        self.debug_color = True

        pyglet.clock.schedule_interval(self.update, 1 / 240.0)
        self.clear()

    def update(self, dt):
        # Avoid the simulation exploding at slow frames by limiting dt
        self.dlgconf.dt = min(dt, 1/100)
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
        self.batch.draw()

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
        if symbol == key.UP:
            self.dlgconf.scale = min(self.dlgconf.scale + 0.05, 1.5)
        if symbol == key.DOWN:
            self.dlgconf.scale = max(self.dlgconf.scale - 0.05, 0.05)
        if symbol == key.Q:
            self.dlgconf.growth = min(self.dlgconf.growth + 0.01, 1.0)
        if symbol == key.W:
            self.dlgconf.growth = max(self.dlgconf.growth - 0.01, 0.0)
        if symbol == key.S:
            self.dlgconf.repulsion = min(self.dlgconf.repulsion + 0.1, 10.0)
        if symbol == key.X:
            self.dlgconf.repulsion = max(self.dlgconf.repulsion - 0.1, 0.1)

    def _update_vertex_list(self):
        centered_path = self.path.get_centered_points(self.width, self.height)
        self.vertex_list.resize(len(self.path))
        self.vertex_list.vertices = list(centered_path.flatten())
        if self.debug_color:
            ones = np.full(len(self.path), 1.0)
            inverscaled_distribution = 1 - (self.path.growth_distribution / self.dlgconf.growth)
            colors = np.ravel(np.column_stack((ones, inverscaled_distribution, inverscaled_distribution, ones)))
            self.vertex_list.colors = colors
        else:
            color = (1, 1, 1, max(0.001, 0.01 * self.dlgconf.scale)) if self.tracing else (1, 1, 1, 1)
            self.vertex_list.colors = color * len(self.path)

    def _update_debug_label(self):
        self.debug_label.text = "Nodes: {}\n".format(len(self.path)) + self.dlgconf.get_multiline_str()


def main():
    simulation_conf = DLGConf(growth=0.005,
                              attraction=15.0,
                              repulsion=1.5,
                              alignement=0.0,
                              perturbation=5.0,
                              repulsion_radius=10.0,
                              min_distance=2.0,
                              max_distance=20.0,
                              scale=1.0)
    window_conf = pyglet.gl.Config(sample_buffers=1, sample=4)
    window = SimulationWindow(dlgconf=simulation_conf, fullscreen=True, config=window_conf)
    pyglet.app.run()


if __name__ == '__main__':
    main()
