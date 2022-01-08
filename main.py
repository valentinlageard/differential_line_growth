import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from algorithm import *


WIDTH = 1500
HEIGHT = 1040


class SimulationWindow(pyglet.window.Window):
    def __init__(self, dlgconf: DLGConf, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dlgconf = dlgconf

        self.batch = pyglet.graphics.Batch()

        self.path = generate_circle(200, 100)
        centered_path = self.path + np.array([self.width / 2, self.height / 2])
        self.vertex_list = self.batch.add(self.path.shape[0], gl.GL_LINE_LOOP, None,
                                          ('v2f', list(centered_path.flatten())),
                                          ('c4f', (1, 1, 1, 0.01) * self.path.shape[0]))

        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.fps_display.label.color = (255, 255, 255, 255)
        self.debug_label = pyglet.text.Label(x=0, y=self.height - 15, width=400, height=400, multiline=True)

        gl.glLineWidth(1)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.playing = False
        self.tracing = False

        pyglet.clock.schedule_interval(self.update, 1 / 120.0)
        self.clear()

    def update(self, dt):
        # Avoid the simulation exploding at slow frames by limiting dt
        self.dlgconf.dt = min(dt, 1/100)
        if self.playing:
            self.path = differential_line_growth(self.path, self.dlgconf)
            self._update_vertex_list()
            self._update_debug_label()

    def on_draw(self):
        if self.playing:
            if not self.tracing:
                self.clear()
                self.fps_display.draw()
            self.batch.draw()
            self.debug_label.draw()

    def on_close(self):
        exit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.playing = not self.playing
        if symbol == key.A:
            self.tracing = not self.tracing
        if symbol == key.R:
            self.clear()
            self.path = generate_circle(200, 100)
        if symbol == key.UP:
            self.dlgconf.scale += 0.1
            if self.dlgconf.scale > 2.0:
                self.dlgconf.scale = 2.0
        if symbol == key.DOWN:
            self.dlgconf.scale -= 0.1
            if self.dlgconf.scale < 0.1:
                self.dlgconf.scale = 0.1

    def _update_vertex_list(self):
        centered_path = self.path + np.array([self.width / 2, self.height / 2])
        self.vertex_list.resize(self.path.shape[0])
        self.vertex_list.vertices = list(centered_path.flatten())
        color = (1, 1, 1, max(0.001, 0.01 * self.dlgconf.scale)) if self.tracing else (1, 1, 1, 1)
        self.vertex_list.colors = color * self.path.shape[0]

    def _update_debug_label(self):
        self.debug_label.text = "Nodes: {}\n".format(self.path.shape[0]) + self.dlgconf.get_multiline_str()


def main():
    simulation_conf = DLGConf(growth=0.2,
                              attraction=20.0,
                              repulsion=3.5,
                              alignement=15.0,
                              perturbation=5.0,
                              repulsion_radius=30.0,
                              min_distance=5.0,
                              max_distance=15.0,
                              scale=1.0)
    window_conf = pyglet.gl.Config(sample_buffers=1, sample=4)
    window = SimulationWindow(dlgconf=simulation_conf, width=WIDTH, height=HEIGHT, config=window_conf)
    pyglet.app.run()


if __name__ == '__main__':
    main()
