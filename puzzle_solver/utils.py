import numpy as np


def get_wrapped_contour_section(contour: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    """
    Get the contour section that lies between start_idx and end_idx, with wraparound if end_idx<start_idx.

    :param contour: the contour.
    :param start_idx: the index of the first point of the section.
    :param end_idx: the index of the point right after the last one in the section.
    """
    if start_idx < end_idx:
        return contour[start_idx:end_idx, :, :]
    return contour[[(idx) % len(contour) for idx in range(start_idx, len(contour) + end_idx)], :, :]


def rotate(points: np.ndarray, angle: float, origin=(0, 0)):
    """
    Rotate a set of points counter-clockwise around an origin.

    :param points: an array of 2d points.
    :param angle: an angle in radians.
    :param origin: The rotations origin.
    """
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    o = np.atleast_2d(origin)
    points = np.atleast_2d(points)
    return np.squeeze((R @ (points.T - o.T) + o.T).T)


def find_contour_idx(contours, point):
    """
    Find the index of the contour containing a point.
    """
    point = np.asarray(point)
    for i in range(len(contours)):
        if (contours[i][:, 0, :] == point).all(axis=-1).any():
            return i
