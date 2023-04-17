from dataclasses import dataclass
from enum import Enum

import numpy as np

from puzzle_solver.utils import rotate


class EdgeType(Enum):
    FLAT = 0
    HEAD = 1
    HOLE = -1


@dataclass
class PuzzleEdge:
    piece_id: int
    piece_num: int  # 0 is up, 1 is left, 2 is down and 3 is right
    coords: np.ndarray
    type: EdgeType
    area: float = 0.0
    _norm: np.ndarray | None = None

    @property
    def norm(self):
        if self._norm is None:
            norm = self.coords[:, 0, :].copy()
            norm -= norm[0]
            angle = np.arctan2(norm[-1, 0], norm[-1, 1])
            norm = rotate(norm, angle)
            if self.type == EdgeType.HOLE:
                norm[:, 0] = -norm[:, 0]
            self._norm = norm
        return self._norm

    @property
    def angle(self):
        return self._norm_angle

    def __repr__(self):
        return f"PuzzleEdge(piece_id={self.piece_id}, piece_num={self.piece_num})"
