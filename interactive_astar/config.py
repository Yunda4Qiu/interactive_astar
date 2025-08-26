from enum import Enum

class GridcellValue(Enum):
    FREE = 0
    OBSTACLE = 1
    PUSH_PENALTY = 2
    # PUSH_PENALTY = 5


# Agent 的 8 个移动方向（上下左右及对角线）
MOVES = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

#（上下左右）
CARDINAL_MOVES = {(1, 0), (-1, 0), (0, 1), (0, -1)}


class MapOriginalValue(Enum):
    FREE = 1
    OBSTACLE = 0
    MOVABLE = 2