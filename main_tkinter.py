import tkinter as tk
from tkinter import Canvas
import time
from algorithm import *


WIDTH = 1500
HEIGHT = 1040


# DRAWING


def draw_path(canvas, path):
    centered_path_plus_one = np.vstack([path, path[0]]) + np.array([WIDTH / 2, HEIGHT / 2])
    canvas.create_line(*(centered_path_plus_one.flatten()),
                       fill='white', smooth=1, splinesteps=0)
    # for i in range(path.shape[0]):
    #     create_center_line(canvas, path[i], path[(i + 1) % len(path)])
    #     canvas.create_oval(path[i][0] - 3 + WIDTH / 2, path[i][1] - 3 + HEIGHT / 2,
    #                        path[i][0] + 3 + WIDTH / 2, path[i][1] + 3 + HEIGHT / 2,
    #                        fill='red')


# INTERACTIONS


playing = True


def toggle_play(*args):
    global playing
    playing = not playing


# MAIN


def main():
    global playing

    app = tk.Tk()
    app.title("Canvas")
    app.bind("<space>", toggle_play)

    canvas = Canvas(app, width=WIDTH, height=HEIGHT)
    canvas.configure(bg='black')
    canvas.pack()

    path = generate_circle(200, 100)

    while True:
        if playing:
            draw_path(canvas, path)

        app.update_idletasks()
        app.update()

        if playing:
            path = differential_growth(path, attraction_strength=0.05, repulsion_strength=0.01, repulsion_radius=5.0,
                                       brownian_strength=0.0, align_strength=0, split_distance=2.0, merge_distance=1.0)
            time.sleep(1/120)
            canvas.delete("all")
            print("Nodes: ", path.shape[0])


if __name__ == '__main__':
    main()