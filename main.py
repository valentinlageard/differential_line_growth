import pyglet
import pyglet.gl as gl
import pyglet.window.key as key
from algorithm import *


WIDTH = 1500
HEIGHT = 1040


class SimulationWindow(pyglet.window.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch = pyglet.graphics.Batch()
        self.path = generate_circle(200, 100)
        centered_path = self.path + np.array([self.width / 2, self.height / 2])
        self.vertex_list = self.batch.add(self.path.shape[0], gl.GL_LINE_LOOP, None,
                                          ('v2f', list(centered_path.flatten())),
                                          ('c4f', (1, 1, 1, 0.01) * self.path.shape[0]))
        self.fps_display = pyglet.window.FPSDisplay(window=self)
        self.fps_display.label.color = (255, 255, 255, 255)

        gl.glLineWidth(1)
        gl.glClearColor(0.0, 0.0, 0.0, 0.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        self.playing = True
        self.tracing = False
        self.scale = 1.0

        pyglet.clock.schedule_interval(self.update, 1 / 120.0)
        self.clear()

    def update(self, dt):
        # Avoid the simulation exploding at slow frames by limiting dt
        dt = min(dt, 1/100)
        if self.playing:
            self.path = differential_growth(self.path,
                                            attraction_strength=20.0 * self.scale * dt,
                                            repulsion_strength=2.0 * self.scale * dt,
                                            repulsion_radius=30.0 * self.scale,
                                            brownian_strength=5.0 * self.scale * dt,
                                            align_strength=20.0 * self.scale * dt,
                                            split_distance=10.0 * self.scale,
                                            merge_distance=5.0 * self.scale)
            centered_path = self.path + np.array([self.width / 2, self.height / 2])
            self.vertex_list.resize(self.path.shape[0])
            self.vertex_list.vertices = list(centered_path.flatten())
            color = (1, 1, 1, max(0.001, 0.01 * self.scale)) if self.tracing else (1, 1, 1, 1)
            self.vertex_list.colors = color * self.path.shape[0]
            print("Nodes: ", self.path.shape[0])

    def on_draw(self):
        if self.playing:
            if not self.tracing:
                self.clear()
                self.fps_display.draw()
            self.batch.draw()

    def on_close(self):
        exit()

    def on_key_press(self, symbol, modifiers):
        if symbol == key.SPACE:
            self.playing = not self.playing
        if symbol == key.A:
            self.tracing = not self.tracing
        if symbol == key.R:
            self.path = generate_circle(200, 100)
        if symbol == key.UP:
            self.scale += 0.1
            if self.scale > 2.0:
                self.scale = 2.0
        if symbol == key.DOWN:
            self.scale -= 0.1
            if self.scale < 0.05:
                self.scale = 0.05



def main():
    config = pyglet.gl.Config(sample_buffers=1, sample=4)
    window = SimulationWindow(width=WIDTH, height=HEIGHT, config=config)
    pyglet.app.run()


if __name__ == '__main__':
    main()
