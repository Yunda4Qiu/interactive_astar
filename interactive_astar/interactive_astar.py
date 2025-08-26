import os
import numpy as np
import heapq

# print the current working directory
print(f"Current working directory: {os.getcwd()}")


from heuristics import manhattan_distance, euclidean_distance, chebyshev_distance, octile_distance
from utils import save_grid, generate_exploration_gif, generate_steps_gif, calculate_path_cost

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


# -------------------------------
# Helper Functions
# -------------------------------

def is_valid_cell(grid, pos):
    """Return True if pos is within boundaries and not an obstacle."""
    x, y = pos
    if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
        return False
    if grid[x, y] == OBSTACLE:
        return False
    return True

def get_move_cost(move):
    """Return cost of a given move based on its type (cardinal or diagonal)."""
    return 1 if abs(move[0]) + abs(move[1]) == 1 else np.sqrt(2)

def try_add_state(new_state, tentative_g, current_state, g_score, f_score,
                    came_from, open_set, h_func, goal, weight):
    """Try inserting the new_state into open_set if it offers a better cost."""
    if new_state not in g_score or tentative_g < g_score[new_state]: 
        came_from[new_state] = current_state
        g_score[new_state] = tentative_g
        h_val = h_func(new_state[0], goal)
        f_score[new_state] = tentative_g + h_val * weight
        heapq.heappush(open_set, (f_score[new_state], new_state))

def reconstruct_path(came_from, current_state):
    """Reconstruct and return the search path from came_from."""
    path = [current_state]
    while current_state in came_from:
        current_state = came_from[current_state]
        path.append(current_state)
    return path[::-1]

# One step push, only in cardinal directions
def get_push_candidates(grid, movable, obj_pos, agent_pos):
    """
    Returns a list of valid push candidate cells for a movable object at obj_pos.
    The candidate cell must be reachable from obj_pos in a cardinal direction,
    not occupied, and not equal to the agent's current position.
    """
    candidates = []
    for delta in CARDINAL_MOVES:
        candidate = (obj_pos[0] + delta[0], obj_pos[1] + delta[1])
        if not is_valid_cell(grid, candidate):
            continue
        if candidate in movable:
            continue
        if candidate == agent_pos:
            continue
        candidates.append(candidate)
    return candidates

# One step pull, only in cardinal directions
def get_pull_candidate(obj_pos, agent_pos, grid, movable):
    """
    Returns agent_pos as a pull candidate if agent and the object's positions
    are cardinally adjacent.
    """
    diff = (agent_pos[0] - obj_pos[0], agent_pos[1] - obj_pos[1])
    if diff in CARDINAL_MOVES and is_valid_cell(grid, agent_pos) and agent_pos not in movable:
        return agent_pos
    return None

# Retraction candidates for pull action in 8-directional movement
def get_retreat_candidates(agent_pos, exclude, grid, movable):
    """
    For a pull action, the agent must retreat from its current cell.
    Returns a list of valid adjacent cells (cardinal directions) that are free,
    not equal to 'exclude', and not already occupied by a movable object.
    """
    candidates = []
    # for delta in CARDINAL_MOVES:
    for delta in MOVES:
        retreat = (agent_pos[0] + delta[0], agent_pos[1] + delta[1])
        if not is_valid_cell(grid, retreat):
            continue
        if retreat == exclude:
            continue
        if retreat in movable:
            continue
        candidates.append(retreat)
    return candidates

def get_push_candidates_diagonal(grid, movable, obj_pos, agent_pos):
    """
    For a movable object at obj_pos, return a list of valid diagonal push candidate cells.
    Four diagonal directions: (1,1), (1,-1), (-1,1), (-1,-1).
    A candidate is accepted if:
      - It lies within grid bounds,
      - It is not an obstacle,
      - It is not already occupied by another movable,
      - And it is not the same as the agent's current cell.
    """
    diagonal_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    diagonal_candidates = []
    for delta in diagonal_directions:
        candidate = (obj_pos[0] + delta[0], obj_pos[1] + delta[1])
        if not is_valid_cell(grid, candidate):
            continue
        if candidate in movable or candidate == agent_pos:
            continue
        diagonal_candidates.append(candidate)
    return diagonal_candidates

def get_pull_candidate_diagonal(obj_pos, agent_pos, grid, movable):
    """
    For a movable object at obj_pos, return agent_pos as a valid diagonal pull candidate
    if the agent is diagonally adjacent. That is, the difference in both coordinates
    must be nonzero and equal to one in absolute value. Also ensures that agent_pos is free.
    """
    diagonal_directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    diff = (agent_pos[0] - obj_pos[0], agent_pos[1] - obj_pos[1])
    if diff in diagonal_directions and is_valid_cell(grid, agent_pos) and agent_pos not in movable:
        return agent_pos
    return None

# This function is not used in the current implementation
# This function only get the candidate for the same direction as the agent move
def get_double_push_candidate(grid, movable, obj_pos, move, agent_pos):
    """
    For a double push action:
      - Starting from an agent moving in direction 'move' into a cell occupied by a movable object (obj_pos),
      - Check if the cell in the same direction (from obj_pos) contains a second movable object.
      - Then, check that the candidate cell beyond the second object is free.
    Returns the candidate cell if valid; otherwise returns None.
    """
    # First object is at obj_pos; second should be right behind it in the same direction.
    second_obj = (obj_pos[0] + move[0], obj_pos[1] + move[1])
    if second_obj not in movable:
        return None
    candidate = (second_obj[0] + move[0], second_obj[1] + move[1])
    if not is_valid_cell(grid, candidate):
        return None
    if candidate in movable:
        return None
    return candidate

def get_double_push_candidates(grid, movable, obj_pos, agent_pos):
    """
    For a double push action:
      - Starting from an agent moving in direction 'move' into a cell occupied by a movable object (obj_pos),
      - Check if the cell in the same direction (from obj_pos) contains a second movable object.
      - Then, check that the candidate cell beyond the second object is free.
    Returns the candidate cell if valid; otherwise returns None.
    """
    candidates = []
    for d2 in MOVES:
        # First object is at obj_pos; second should be next to it in direction d2 (not necessarily the same as agent's move).
        second_obj = (obj_pos[0] + d2[0], obj_pos[1] + d2[1])
        if not is_valid_cell(grid, second_obj): # check boundaries and obstacles
            continue
        if second_obj not in movable:
            continue
        if second_obj == agent_pos:
            continue
        second_obj_distance = np.sqrt((second_obj[0] - agent_pos[0]) ** 2 + (second_obj[1] - agent_pos[1]) ** 2)
        if second_obj_distance <= 1:
            continue
        # Check if the candidate cell beyond the second object is free.
        candidate = (second_obj[0] + d2[0], second_obj[1] + d2[1])
        if not is_valid_cell(grid, candidate):
            continue
        if candidate in movable:
            continue
        candidates.append((candidate, d2))
    return candidates


#####################
# Interactive A* Search Algorithm
#####################

# def astar_search(grid, start, goal, movable_objects, distance_func='euclidean',
#                  register_explored=False, filter_radius=5, weight=1,
#                  enable_diagonal_push_pull=False, enable_double_push=False):
def interactive_astar(grid, start, goal, movable_objects, distance_func='euclidean',
                 register_explored=False, filter_radius=5, weight=1,
                 enable_diagonal_push_pull=False, enable_double_push=False):
    """
    A* search with support for (optionally) diagonal push/pull and double push.
    """
    start_state = (start, frozenset(movable_objects))
    open_set = []
    heapq.heappush(open_set, (0, start_state)) # (f_score, state)
    came_from = {}
    g_score = {start_state: 0}

    heuristic_funcs = {
        'manhattan': manhattan_distance,
        'euclidean': euclidean_distance,
        'chebyshev': chebyshev_distance,
        'octile': octile_distance
    }
    assert distance_func in heuristic_funcs, f"Undefined distance function: {distance_func}"
    h_func = heuristic_funcs[distance_func]
    f_score = {start_state: h_func(start, goal)} # f_score = g_score + h(start, goal)

    explored_states = []
    closed_set = set()

    while open_set:
        current_f, current_state = heapq.heappop(open_set)
        if current_state in closed_set:
            continue
        closed_set.add(current_state)

        current_pos, current_movable = current_state
        if register_explored:
            explored_states.append(current_state)

        if current_pos == goal:
            path = reconstruct_path(came_from, current_state)
            total_cost = g_score[current_state]
            return path, explored_states, total_cost

        for move in MOVES:
            new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
            if not is_valid_cell(grid, new_pos): # Check boundaries and obstacles
                continue

            move_cost = get_move_cost(move)
            new_movable = set(current_movable)

            # If stepping into a movable object: handle push/pull
            if new_pos in current_movable:
                # --- Double Push ---
                if enable_double_push:
                    double_push_candidates = get_double_push_candidates(grid, current_movable, new_pos, current_pos)
                    for candidate, d2 in double_push_candidates:
                        first_obj = new_pos
                        second_obj = (first_obj[0] + d2[0], first_obj[1] + d2[1])
                        target_first = second_obj
                        target_second = candidate
                        new_movable_candidate = set(current_movable)
                        # remove original positions
                        new_movable_candidate.remove(first_obj)
                        new_movable_candidate.remove(second_obj)
                        # add updated positions
                        new_movable_candidate.add(target_first)
                        new_movable_candidate.add(target_second)

                        obj_push2_distance = get_move_cost(d2)
                        candidate_cost = move_cost + 2 * PUSH_PENALTY * obj_push2_distance
                        tentative_g = g_score[current_state] + candidate_cost
                        new_state = (new_pos, frozenset(new_movable_candidate))
                        try_add_state(new_state, tentative_g, current_state,
                                    g_score, f_score, came_from,
                                    open_set, h_func, goal, weight)

                # --- Single Push ---
                if enable_diagonal_push_pull:
                    push_candidates = (
                        get_push_candidates(grid, current_movable, new_pos, current_pos)
                        + get_push_candidates_diagonal(grid, current_movable, new_pos, current_pos)
                    )
                else:
                    push_candidates = get_push_candidates(grid, current_movable, new_pos, current_pos)
                for candidate in push_candidates:
                    new_movable_candidate = set(current_movable)
                    new_movable_candidate.remove(new_pos)
                    new_movable_candidate.add(candidate)
                    obj_push_distance = get_move_cost((candidate[0] - new_pos[0], candidate[1] - new_pos[1]))
                    candidate_cost = move_cost + PUSH_PENALTY * obj_push_distance
                    tentative_g = g_score[current_state] + candidate_cost
                    new_state = (new_pos, frozenset(new_movable_candidate))
                    try_add_state(new_state, tentative_g, current_state,
                                  g_score, f_score, came_from,
                                  open_set, h_func, goal, weight)

                # --- Pull ---
                pull_candidates = []
                if enable_diagonal_push_pull:
                    c1 = get_pull_candidate(new_pos, current_pos, grid, current_movable)
                    c2 = get_pull_candidate_diagonal(new_pos, current_pos, grid, current_movable)
                    if c1: pull_candidates.append(c1)
                    if c2: pull_candidates.append(c2)
                else:
                    c = get_pull_candidate(new_pos, current_pos, grid, current_movable)
                    if c: pull_candidates.append(c)

                for pull_cell in pull_candidates:
                    new_movable_candidate = set(current_movable)
                    new_movable_candidate.remove(new_pos)
                    new_movable_candidate.add(current_pos)
                    for retreat in get_retreat_candidates(current_pos, new_pos, grid, current_movable):
                        obj_pull_distance = get_move_cost((current_pos[0] - new_pos[0], current_pos[1] - new_pos[1]))
                        pull_cost = move_cost + PUSH_PENALTY * obj_pull_distance
                        tentative_g = g_score[current_state] + pull_cost
                        new_state = (retreat, frozenset(new_movable_candidate))
                        try_add_state(new_state, tentative_g, current_state,
                                      g_score, f_score, came_from,
                                      open_set, h_func, goal, weight)
                continue  # skip normal moves after handling push/pull

            # Standard move into free cell
            new_state = (new_pos, frozenset(new_movable))
            tentative_g = g_score[current_state] + move_cost
            try_add_state(new_state, tentative_g, current_state,
                          g_score, f_score, came_from,
                          open_set, h_func, goal, weight)

    return None, explored_states, None



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
    movable_objects = [(2, 5), (3, 1), (4, 1), (5, 1), (5, 3), (6, 1), (6, 7)]
    # Define start and goal positions.
    start = (1, 1)
    goal = (8, 6)

    path, explored_states, total_cost = interactive_astar(
    grid, start, goal, movable_objects,
    distance_func='euclidean',
    register_explored=True,
    filter_radius=5,
    weight=1,
    enable_diagonal_push_pull=True,  # Enable diagonal push/ pull
    enable_double_push=True)         # Enable double push action
    
    save_dir = 'results/astar_test/'
    map_name = 'astar_test_map'
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
    print(f'Exploration length: {len(explored_states)}')
    generate_steps_gif(grid, goal, path, f"{file_path}_steps.gif")

if __name__ == "__main__":
    main()