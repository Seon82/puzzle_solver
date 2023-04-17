import warnings

import cv2
import numpy as np

from puzzle_solver.edge import EdgeType
from puzzle_solver.utils import rotate


class PuzzlePiece:
    def __init__(self, img, contour, edges, id):
        contour = contour.copy()
        x, y, w, h = cv2.boundingRect(contour)
        cropped = img[y : y + h, x : x + w]
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        contour[:, 0, 0] -= x
        contour[:, 0, 1] -= y
        cv2.fillPoly(mask, [contour], 1)

        self.id = id
        self.img = cv2.bitwise_and(cropped, cropped, mask=mask)
        self.edges = sorted(edges, key=lambda x: x.piece_num)
        self.rot = 0  # Number of counter-clockwise 90° rotations
        self.corner = sum(e.type == EdgeType.FLAT for e in self.edges) == 2
        self._was_rotated = False

    def rotate(self, n=None, force=False):
        if self._was_rotated and not force:
            warnings.warn("This puzzle piece had already been rotate. Pass force=True to rotate again anyway")
            return
        if n is None:
            n = self.rot
        for _ in range(n):
            self.img = np.rot90(self.img)
        self._was_rotated = True

    def place(self, img, pos):
        x, y = pos
        h, w = self.img.shape[:2]
        img[y : y + h, x : x + w] = cv2.add(img[y : y + h, x : x + w], self.img)
        return img

    def __repr__(self):
        return f"PuzzlePiece"
