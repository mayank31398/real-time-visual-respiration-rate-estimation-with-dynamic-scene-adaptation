import cv2.cv2 as cv2
import matplotlib.animation as anime
import matplotlib.pyplot as plt

import numpy as np
from module import Computeflow, GetImage


class LiveGraph:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = plt.subplot(111)
        self.capture = cv2.VideoCapture(0)
        self.scale = (120, 160)
        self.window = 100
        self.lambda_ = 0.8
        self.ma = 25
        self.respiration_signal = [0]
        self.image_old = GetImage(self.capture, self.scale)
        self.image_new = self.image_old
        self.tflow = Computeflow(self.image_new, self.image_old, (30, 40))
        self.dflow = self.tflow
        self.frame = 0

    def Animate(self, i):
        if(self.frame > 0):
            self.image_new = GetImage(self.capture, self.scale)

        flow = Computeflow(self.image_new, self.image_old, (30, 40))
        self.image_old = self.image_new

        if(self.frame != 0):
            self.tflow = self.lambda_ * self.tflow + flow
            magnitude = np.linalg.norm(self.tflow)

            if(magnitude > self.ma):
                self.tflow = self.tflow * self.ma / magnitude

            if((self.dflow * flow).sum() > 0):
                self.dflow = self.lambda_ * self.dflow + flow
            else:
                self.dflow = self.lambda_ * self.dflow - flow

            magnitude = np.linalg.norm(self.dflow)
            if(magnitude > self.ma):
                self.dflow = self.dflow * self.ma / magnitude
                magnitude = self.ma

            respiration_velocity = (self.dflow * self.tflow).sum() / magnitude
            if(not np.isnan(respiration_velocity)):
                if(self.frame < self.window):
                    n = self.frame
                else:
                    n = self.window
                r = 0.9 * self.respiration_signal[n - 1] + respiration_velocity
            else:
                r = 0

            if(self.frame > self.window):
                self.respiration_signal = self.respiration_signal[1:]
            self.respiration_signal.append(r)

        self.frame += 1

        self.ax.clear()
        self.ax.plot(self.respiration_signal)


def main():
    graph = LiveGraph()
    animation = anime.FuncAnimation(graph.fig, graph.Animate, interval=1)
    plt.show()


if(__name__ == "__main__"):
    main()
