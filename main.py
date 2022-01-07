import pyglet
import pyglet.gl as gl
from algorithm import *

WIDTH = 1500
HEIGHT = 1040


class SimulationWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch = pyglet.graphics.Batch()
        self.path = generate_circle(200, 100)
        centered_path = self.path + np.array([WIDTH / 2, HEIGHT / 2])
        self.vertex_list = self.batch.add(self.path.shape[0], gl.GL_LINE_LOOP, None,
                                          ('v2f', list(centered_path.flatten())),
                                          ('c4f', (1, 1, 1, 0.01) * self.path.shape[0]))
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.fps_display.label.color = (255, 255, 255, 255)

        gl.glLineWidth(1)
        # gl.glClearColor(0.0, 0.0, 0.1, 0.9)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.playing = True  # TODO: to use for play/pause interaction

        pyglet.clock.schedule_interval(self.update, 1 / 120.0)
        self.clear()

    def update(self, dt):
        # TODO: use dt !
        if self.playing:
            self.path = differential_growth(self.path,
                                            attraction_strength=0.2, repulsion_strength=0.018, repulsion_radius=30.0,
                                            brownian_strength=0.09, align_strength=0.01, split_distance=10.0,
                                            merge_distance=5.0)
            centered_path = self.path + np.array([WIDTH / 2, HEIGHT / 2])
            self.vertex_list.resize(self.path.shape[0])
            self.vertex_list.vertices = list(centered_path.flatten())
            self.vertex_list.colors = (1, 1, 1, 0.005) * self.path.shape[0]
            print("Nodes: ", self.path.shape[0])

    def on_draw(self):
        # self.clear()
        # self.fps_display.draw()
        if self.playing:
            self.batch.draw()

    def on_close(self):
        self.close()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            self.playing = not self.playing


def main():
    config = pyglet.gl.Config(sample_buffers=1, sample=4)
    window = SimulationWindow(width=WIDTH, height=HEIGHT, config=config)
    try:
        pyglet.app.run()
    except Exception as e:
        print("Exception caught: ", e)


if __name__ == '__main__':
    main()
