import os

import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend("QT5Agg")

EPSILON = 1e-10

class Queue:
    def __init__(self, capacity=50):
        self.array = []
        self.capacity = capacity
    
    def Enqueue(self, x):
        if(len(l) > capacity):
            self.Dequeue()
        self.l.append(x)
    
    def Dequeue(self):
        x = self.l[0]
        self.l = self.l[1:]
        return x


def Display(array):
    print(array)
    # array = array.tolist()
    # for i in array:
    #     print(i)
    # exit()


def GetRespirationRate(
    image_n_1: np.ndarray,
    image_n: np.ndarray,
    F_bar_n_1: np.ndarray,
    V_n_1: np.ndarray,
    v=0.95,
    mu=0.95
):
    def ComputeSpatialDerivative(image, axis):
        derivative = np.diff(image, axis=axis)
        if(axis == 0):
            padding = np.zeros((1, derivative.shape[1]), dtype=np.float32)
        else:
            padding = np.zeros((derivative.shape[0], 1), dtype=np.float32)
        derivative = np.concatenate([padding, derivative], axis=axis)
        return derivative

    image_n = image_n.astype(np.float32)
    image_n_1 = image_n_1.astype(np.float32)

    # 1st step
    D_n = image_n - image_n_1  # m, n

    # 2nd step
    dx = ComputeSpatialDerivative(image_n, axis=1)  # m, n
    dy = ComputeSpatialDerivative(image_n, axis=0)  # m, n

    dx = np.expand_dims(dx, axis=2)
    dy = np.expand_dims(dy, axis=2)

    G_n = np.concatenate([dx, dy], axis=2)  # m, n, 2

    # 3rd step
    D_n = D_n / (np.linalg.norm(G_n, axis=2) ** 2 + EPSILON)
    D_n = np.expand_dims(D_n, axis=2)
    f_n = -D_n * G_n  # m, n, 2

    # 4th step
    F_n = f_n.reshape(-1)  # 2mn

    # 5th step
    F_bar_n = v * F_bar_n_1 + F_n

    # 6th step
    sgn = np.sign(np.dot(F_n, V_n_1))
    V_n = mu * V_n_1 + sgn * F_n

    # 7th sign
    r_n = np.dot(V_n, F_bar_n) / (np.linalg.norm(V_n) + EPSILON)

    return r_n, F_bar_n, V_n


# TODO
def main():
    cap = cv2.VideoCapture(0)
    ret, frame_ = cap.read()
    frame_ = cv2.cvtColor(frame_, cv2.COLOR_BGR2GRAY)

    l = []

    resolution = frame_.shape

    F_bar_n = np.random.randn(2 * resolution[0] * resolution[1])
    V_n = np.random.randn(2 * resolution[0] * resolution[1])

    while(True):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        r_n, F_bar_n, V_n = GetRespirationRate(frame_, frame, F_bar_n, V_n)
        
        l.append(r_n)
        print(len(l), r_n)

        frame_ = frame

        if(len(l) == 1000):
            break
    
    plt.plot(l)
    plt.show()


if(__name__ == "__main__"):
    main()
