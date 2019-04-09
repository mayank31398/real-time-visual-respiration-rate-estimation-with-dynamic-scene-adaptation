import cv2
import matplotlib.pyplot as plt
import numpy as np


def Computeflow(image_new, image_old, scale):
    def ComputeSpatialDerivative(image, axis):
        derivative = np.diff(image, axis=axis)
        if(axis == 0):
            padding = np.zeros((1, derivative.shape[1]), dtype=np.float32)
        else:
            padding = np.zeros((derivative.shape[0], 1), dtype=np.float32)
        derivative = np.concatenate([padding, derivative], axis=axis)
        return derivative

    dx = ComputeSpatialDerivative(image_new, axis=1)
    dy = ComputeSpatialDerivative(image_new, axis=0)
    dx = np.expand_dims(dx, axis=2)
    dy = np.expand_dims(dy, axis=2)

    gradients = np.concatenate([dx, dy], axis=2)
    magnitude = (gradients ** 2).sum(axis=2)
    magnitude[magnitude < 9] = np.Inf
    ImageDiff = image_old - image_new

    flowField = ImageDiff / magnitude
    flowField = gradients * np.expand_dims(flowField, axis=2)
    flowField = cv2.resize(flowField, scale)

    return flowField


def GetImage(cap, scale):
    frame = cap.read()[1] / 2 + cap.read()[1] / 4 + cap.read()[1] / 2
    frame = frame[..., 0] * 0.29 + frame[..., 1] * 0.59 + frame[..., 2] * 0.11
    frame = cv2.GaussianBlur(frame, (25, 25), 2)
    frame = cv2.resize(frame, scale)
    return frame


def main():
    with open("output.txt", "w") as file:
        capture = cv2.VideoCapture(0)
        scale = (120, 160)
        N = 100
        lambda_ = 0.8
        MA = 25
        respiration_signal = [0]

        image_old = GetImage(capture, scale)
        image_new = image_old

        for i in range(N):
            if(i > 0):
                image_new = GetImage(capture, scale)
            
            flow = Computeflow(image_new, image_old, (30, 40))
            image_old = image_new

            if(i == 0):
                tflow = flow
                dflow = flow
            else:
                tflow = lambda_ * tflow + flow
                magnitude = np.linalg.norm(tflow)

                if(magnitude > MA):
                    tflow = tflow * MA / magnitude

                if((dflow * flow).sum() > 0):
                    dflow = lambda_ * dflow + flow
                else:
                    dflow = lambda_ * dflow - flow

                magnitude = np.linalg.norm(dflow)
                if(magnitude > MA):
                    dflow = dflow * MA / magnitude
                    magnitude = MA

                respiration_velocity = (dflow * tflow).sum() / magnitude
                if(not np.isnan(respiration_velocity)):
                    respiration_signal.append(
                        0.9 * respiration_signal[i - 1] + respiration_velocity)
                    file.write(str(respiration_signal[len(respiration_signal) - 1]) + "\n")
                else:
                    respiration_signal.append(0)
                    file.write("0\n")


if(__name__ == "__main__"):
    main()
