import matplotlib.pyplot as plt
import numpy as np


def plot(img: np.array, figsize=None):
    if figsize is not None:
        plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap="gray", interpolation="none")
    else:
        plt.imshow(img[:, :, ::-1])
