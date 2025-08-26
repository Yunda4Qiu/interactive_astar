import numpy as np

def manhattan_distance(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def euclidean_distance(pos, goal):
    return np.sqrt((pos[0] - goal[0]) ** 2 + (pos[1] - goal[1]) ** 2)

def chebyshev_distance(pos, goal):
    return max(abs(pos[0] - goal[0]), abs(pos[1] - goal[1]))

def octile_distance(pos, goal):
    dx = abs(pos[0] - goal[0])
    dy = abs(pos[1] - goal[1])
    return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)
    