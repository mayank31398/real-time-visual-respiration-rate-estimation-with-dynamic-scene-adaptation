import os

import cv2.cv2 as cv2
import numpy as np


def GetRespirationRate(
    image_n: np.ndarray,
    image_n_1: np.ndarray,
    F_n_1: np.ndarray,
    V_n_1: np.ndarray,
    v=0.95,
    mu=0.95
):
    def GetSign(x):
        if(x > 0):
            return 1
        elif(x < 0):
            return -1
        return 0

    def ComputeSpatialDerivative(image, axis):
        derivative = np.diff(image, axis=axis)
        padding = np.zeros((derivative.shape[0], 1))
        if(axis == 0):
            padding = padding.T
        derivative = np.concatenate([padding, image], axis=axis)
        return derivative

    # 1st step
    D_n = image_n - image_n_1

    # 2nd step
    dx = ComputeSpatialDerivative(image_n, axis=1)
    dy = ComputeSpatialDerivative(image_n, axis=0)

    G_n = np.concatenate([dx, dy], axis=0)

    # 3rd step
    f_n = -D_n * G_n / np.linalg.norm(G_n) ** 2

    # 4th step
    F_n = f_n.reshape(-1)

    # 5th step
    F_n = v * F_n_1 + F_n

    # 6th step
    sgn = GetSign(np.dot(F_n, V_n_1))
    V_n = mu * V_n_1 + sgn * F_n

    # 7th sign
    r_n = np.dot(V_n, F_n) / np.linalg.norm(V_n)

    return r_n, F_n, V_n


# TODO
def main():
    count = 0
    success = 1

    video = cv2.VideoCapture(os.path.join("Data", "video"))
    while(success):
        success, image = video.read()
    
    