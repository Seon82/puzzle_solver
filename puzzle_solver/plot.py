import matplotlib.pyplot as plt
import numpy as np


def plot(img: np.array):
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray", interpolation="none")
    else:
        plt.imshow(img[:, :, ::-1])
