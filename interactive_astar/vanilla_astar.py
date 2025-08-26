import os
import numpy as np
import heapq

# print the current working directory
print(f"Current working directory: {os.getcwd()}")


from utils2 import GridBuilder, show_grid
from heuristics import manhattan_distance, euclidean_distance, chebyshev_distance, octile_distance
from utils import save_grid, generate_exploration_gif, generate_steps_gif, calculate_path_cost
from astar import is_valid_cell, get_move_cost, reconstruct_path

# ----------------------------------------
# Parameter settings and constants
# ----------------------------------------
from config import GridcellValue

FREE = GridcellValue.FREE.value                 # 0, free space
OBSTACLE = GridcellValue.OBSTACLE.value         # 1, obstacle
PUSH_PENALTY = GridcellValue.PUSH_PENALTY.value # 2, extra penalty when pushing a movable object

# Eight possible moves for the agent (4 cardinal + 4 diagonal)
MOVES = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]

# Movable objects can be pushed or pulled only in cardinal directions
CARDINAL_MOVES = {(1, 0), (-1, 0), (0, 1), (0, -1)}



def vanilla_astar(grid, start, goal, movable_objects, distance_func='euclidean', weight=1):
    """
    Classic A* pathfinding that avoids both static (OBSTACLE) and movable obstacles.
    Supports 8-directional movement.

    Parameters:
        grid: 2D numpy array
        start: (x, y) tuple
        goal: (x, y) tuple
        movable_objects: list of (x, y) positions to avoid
        distance_func: heuristic name
        weight: heuristic weight

    Returns:
        path: list of (pos, movable) tuples
        explored: list of (pos, movable) tuples
        total_cost: accumulated cost
    """
    heuristic_funcs = {
        'manhattan': manhattan_distance,
        'euclidean': euclidean_distance,
        'chebyshev': chebyshev_distance,
        'octile': octile_distance
    }
    assert distance_func in heuristic_funcs, f"Invalid heuristic: {distance_func}"
    h_func = heuristic_funcs[distance_func]

    start_state = start
    open_set = []
    heapq.heappush(open_set, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}
    f_score = {start_state: h_func(start, goal) * weight}
    explored = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in explored:
            continue
        explored.add(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            path_with_state = [(pos, frozenset(movable_objects)) for pos in path]
            explored_with_state = [(pos, frozenset(movable_objects)) for pos in explored]
            return path_with_state, explored_with_state, g_score[current]

        for move in MOVES:
            neighbor = (current[0] + move[0], current[1] + move[1])
            if not is_valid_cell(grid, neighbor):
                continue
            if neighbor in movable_objects:
                continue

            move_cost = get_move_cost(move)
            tentative_g = g_score[current] + move_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + h_func(neighbor, goal) * weight
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None, [], None  # No path found



# -------------------------------
# Main Execution Function
# -------------------------------

def main():
    # Define grid size.
    rows, cols = 10, 10
    grid = np.zeros((rows, cols), dtype=int)

    # Set boundary obstacles.
    grid[0, :] = OBSTACLE
    grid[rows - 1, :] = OBSTACLE
    grid[:, 0] = OBSTACLE
    grid[:, cols - 1] = OBSTACLE

    # Define internal obstacles.
    obstacles = [(3, i) for i in range(2, 8)]
    obstacles += [(4, 7), (5, 6), (5, 7)]
    obstacles += [(7, i) for i in range(4, 9)]
    obstacles += [(6, 8)]
    obstacles += [(4, 2), (5, 2), (6, 2), (6, 6)]
    for pos in obstacles:
        grid[pos] = OBSTACLE

    # Define movable objects.
    # movable_objects = [(2, 5), (3, 1), (4, 1), (5, 1), (5, 3), (6, 1), (6, 7)]
    movable_objects = [(2, 5), (3, 1), (4, 1), (5, 1), (5, 3), (6, 1)]

    # Define start and goal positions.
    start = (1, 1)
    goal = (8, 6)

    # path, explored_states, total_cost = astar_search(
    #                                     grid, start, goal, movable_objects,
    #                                     distance_func='euclidean',
    #                                     register_explored=True,
    #                                     filter_radius=5,
    #                                     weight=1,
    #                                     enable_diagonal_push_pull=True,  # Enable diagonal push/ pull
    #                                     enable_double_push=True)         # Enable double push action
    path, explored_states, total_cost = vanilla_astar(
                                        grid, start, goal, movable_objects,
                                        distance_func='euclidean',
                                        weight=1
)
    save_dir = 'results/vanilla_astar_test/'
    map_name = 'vanilla_astar_test_map_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, map_name)

    save_grid(grid, start, goal, movable_objects, f"{file_path}.png")

    if path is None:
        print("Path not found.")
        return
    
    path_length, path_cost = calculate_path_cost(path=path)
    print(f'Path length: {path_length}, \nPath cost: {path_cost}')
    print(f'Total cost (including PUSH_PENALTY): {total_cost}')
    for i, p in enumerate(path):
        print(f'Step {i+1}: {p[0]}')
    # generate_exploration_gif(grid, goal, explored_states, path, f"{file_path}_exploration.gif")
    # print(f'Exploration length: {len(explored_states)}')
    # generate_steps_gif(grid, goal, path, f"{file_path}_steps.gif")


def main2():
    OBSTACLE = GridcellValue.OBSTACLE.value

    builder = GridBuilder(
        rows=10, cols=10,
        base_obstacles=[(3,i) for i in range(2,8)] + [(7, i) for i in range(5, 10)] + [(4, 7), (5, 7)],
        # base_movables=[(2, 5), (3, 1),(3, 8), (4, 1), (5, 3)],
        base_movables=[(2, 5), (3, 1), (5, 3)],
        start=(1, 1), goal=(8, 6)
        )
    grid_7_added_obstacles = [(4, 2), (5, 2), (6, 2)] \
                         + [(5, 6), (2, 8)]
    # grid_7_added_movables = [(5, 1), (6, 1), (6, 6), (6, 7)]
    grid_7_added_movables = []
    grid_7, mov_7, start_7, goal_7 = builder.build(add_obs=grid_7_added_obstacles, 
                                                add_mov=grid_7_added_movables)
    show_grid(grid_7, start_7, goal_7, mov_7)

    save_dir = 'results/vanilla_astar_test/'
    map_name = 'vanilla_astar_test_map_2'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, map_name)
    save_grid(grid_7, start_7, goal_7, mov_7, f"{file_path}.png")

    path, explored_states, total_cost = vanilla_astar(
                                    grid_7, start_7, goal_7, mov_7,
                                    distance_func='euclidean',
                                    weight=1
                                    )

    if path is None:
        print("Path not found.")
        return

    path_length, path_cost = calculate_path_cost(path=path)
    print(f'Path length: {path_length}, \nPath cost: {path_cost}')
    print(f'Total cost (including PUSH_PENALTY): {total_cost}')
    for i, p in enumerate(path):
        print(f'Step {i+1}: {p[0]}')
    # generate_exploration_gif(grid, goal, explored_states, path, f"{file_path}_exploration.gif")
    # print(f'Exploration length: {len(explored_states)}')
    # generate_steps_gif(grid, goal, path, f"{file_path}_steps.gif")


if __name__ == "__main__":
    # main()
    main2()
