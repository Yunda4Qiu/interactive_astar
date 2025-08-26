import os
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from config import GridcellValue


import matplotlib.patheffects as pe
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import defaultdict

import math

# ----------------------------------------
# Global color configuration
# ----------------------------------------
COLOR_OBSTACLE = 'black'
COLOR_FREE     = 'white'
COLOR_EXPLORED = 'lightblue'
COLOR_PATH     = 'orange'
COLOR_MOVABLE  = 'blue'
COLOR_AGENT    = 'green'
COLOR_AGENT_STEP = 'orange'
COLOR_GOAL     = 'red' # 

# ----------------------------------------
# Grid value constants
# ----------------------------------------
OBSTACLE = GridcellValue.OBSTACLE.value
FREE     = GridcellValue.FREE.value

# ----------------------------------------
# Drawing functions
# ----------------------------------------
def draw_grid(grid, agent, goal, movable, explored=None, path=None, title=""):
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(int(cols*0.4), int(rows*0.4)))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.invert_yaxis()
    
    for i in range(rows):
        for j in range(cols):
            color = COLOR_OBSTACLE if grid[i, j] == OBSTACLE else COLOR_FREE
            rect = Rectangle((j, i), 1, 1, facecolor=color)
            ax.add_patch(rect)
    
    if explored:
        for state in explored:
            pos, _ = state
            circle = plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.2, color=COLOR_EXPLORED, alpha=0.5)
            ax.add_patch(circle)

    if path:
        path_positions = [state[0] for state in path]
        for pos in path_positions:
            rect = Rectangle((pos[1], pos[0]), 1, 1, facecolor=COLOR_PATH, alpha=0.5)
            ax.add_patch(rect)

    for m in movable:
        circle = plt.Circle((m[1] + 0.5, m[0] + 0.5), 0.3, color=COLOR_MOVABLE)
        ax.add_patch(circle)

    circle = plt.Circle((agent[1] + 0.5, agent[0] + 0.5), 0.3, color=COLOR_AGENT)
    ax.add_patch(circle)

    circle = plt.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.3, color=COLOR_GOAL)
    ax.add_patch(circle)

    ax.set_xticks(np.arange(0, cols + 1, 1))
    ax.set_yticks(np.arange(0, rows + 1, 1))
    ax.grid(True, color='gray', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()
    return fig, ax


def fig_to_image(fig):
    fig.canvas.draw()
    width, height = fig.get_size_inches() * fig.dpi
    width, height = int(width), int(height)
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))[:, :, :3]
    plt.close(fig)
    return image


def generate_exploration_gif(grid, goal, explored_states, final_path, output_file):
    frames = []
    num_frames = 20
    step_interval = max(1, len(explored_states) // num_frames)
    for i in range(0, len(explored_states), step_interval):
        explored_subset = explored_states[:i + 1]
        current_state = explored_states[i]
        agent, movable = current_state
        fig, ax = draw_grid(grid, agent, goal, movable, explored_subset, title=f"Explorate Step {i}")
        frame = fig_to_image(fig)
        frames.append(frame)

    final_state = final_path[-1]
    agent, movable = final_state
    fig, ax = draw_grid(grid, agent, goal, movable, explored_states, final_path, title="Final Path")
    frame = fig_to_image(fig)
    frames.append(frame)

    imageio.mimsave(output_file, frames, fps=3)
    print(f"Exploration GIF Save to: {output_file}")


def generate_steps_gif(grid, goal, path, output_file):
    frames = []
    for idx, state in enumerate(path):
        agent, movable = state
        partial_path = path[:idx + 1]
        fig, ax = draw_grid(grid, agent, goal, movable, path=partial_path, title=f"Step: {idx}")
        frame = fig_to_image(fig)
        frames.append(frame)
    imageio.mimsave(output_file, frames, fps=3)
    print(f"Steps GIF save to: {output_file}")


def show_grid(grid, start, goal, movable):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.scatter([m[1] for m in movable], [m[0] for m in movable], color=COLOR_MOVABLE, s=100)
    ax.scatter(start[1], start[0], marker='o', color=COLOR_AGENT, s=80)
    ax.scatter(goal[1], goal[0], marker='*', color=COLOR_GOAL, s=80)
    plt.show()


def save_grid(grid, start, goal, movable, output_file, dpi=150):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.scatter([m[1] for m in movable], [m[0] for m in movable], color=COLOR_MOVABLE, s=100)
    ax.scatter(start[1], start[0], marker='o', color=COLOR_AGENT, s=80)
    ax.scatter(goal[1], goal[0], marker='*', color=COLOR_GOAL, s=80)
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Grid image saved to: {output_file}")
    plt.close(fig)


def save_grid_with_legend(grid, start, goal, movable, output_file, dpi=150):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    # 绘制起点、终点、movable物体
    ax.scatter([m[1] for m in movable], [m[0] for m in movable], color='blue', s=100)
    ax.scatter(start[1], start[0], marker='o', color='green', s=80)
    ax.scatter(goal[1], goal[0], marker='*', color='red', s=100)

    legend_elements = []
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', label='Start',
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Goal',
               markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Movable',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='obstacle',
               markerfacecolor='grey', markersize=10)
    ]

    # # 创建 legend patches
    # start_patch = mpatches.Circle((0, 0), radius=5, color='green', label='Start')
    # goal_patch = mpatches.RegularPolygon((0, 0), numVertices=5, radius=5, color='red', label='Goal')
    # movable_patch = mpatches.Rectangle((0, 0), 1, 1, color='blue', label='Movable')
    # obstacle_patch = mpatches.Rectangle((0, 0), 1, 1, color='black', label='Obstacle')

    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"Grid image saved to: {output_file}")
    plt.close(fig)

def calculate_path_cost(path):
    path_length = len(path)
    path_position = [x[0] for x in path]
    total_cost = 0
    for i in range(path_length - 1):
        dx = abs(path_position[i + 1][0] - path_position[i][0])
        dy = abs(path_position[i + 1][1] - path_position[i][1])
        cost = np.sqrt(dx + dy)
        total_cost += cost
    return path_length, total_cost


def print_path_and_exploration_length(path, exploration):
    if path is not None:
        print(f"Path's length: {len(path)}")
        print(f"Exploration's length: {len(exploration)}")


def show_path(grid, path):
    start_pos = path[0][0]
    goal_pos = path[-1][0]
    movable_pos = path[-1][1]
    path_pos = [x[0] for x in path]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.scatter([p[1] for p in path_pos], [p[0] for p in path_pos], color=COLOR_PATH, s=100)
    ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color=COLOR_MOVABLE, s=100)
    ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=80)
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=80)
    plt.show()


def save_path(grid, path, output_file, dpi=150):
    start_pos = path[0][0]
    goal_pos = path[-1][0]
    movable_pos = path[-1][1]
    path_pos = [x[0] for x in path]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.scatter([p[1] for p in path_pos], [p[0] for p in path_pos], color=COLOR_PATH, s=100)
    ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color=COLOR_MOVABLE, s=100)
    ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=80)
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=80)

    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def show_every_step(grid, path):
    num_steps = len(path)
    cols = min(num_steps, 5)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(-1)

    start_pos = path[0][0]
    goal_pos = path[-1][0]

    for i, (pos, movable_pos) in enumerate(path):
        ax = axes[i]
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color=COLOR_MOVABLE, s=60)
        ax.scatter(pos[1], pos[0], color=COLOR_AGENT_STEP, s=80, marker='o')
        ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=40)
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=60)
        ax.set_title(f"Step {i}")

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def save_every_step(grid, path, output_file, max_cols=5, dpi=150):
    num_steps = len(path)
    cols = min(num_steps, max_cols)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(-1)

    start_pos = path[0][0]
    goal_pos  = path[-1][0]

    for i, (pos, movable_pos) in enumerate(path):
        ax = axes[i]
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color=COLOR_MOVABLE, s=60)
        ax.scatter(pos[1], pos[0], color=COLOR_AGENT_STEP, s=80, marker='o')
        ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=40)
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=60)
        ax.set_title(f"Step {i}")

    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


class GridBuilder:
    def __init__(self, rows, cols, base_obstacles, base_movables, start, goal):
        self.rows = rows
        self.cols = cols
        self.base_obs = set(base_obstacles)
        self.base_mov = set(base_movables)
        self.start = start
        self.goal  = goal

    def build(self, add_obs=None, del_obs=None, add_mov=None, del_mov=None):
        grid = np.zeros((self.rows, self.cols), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OBSTACLE
        obs = (self.base_obs | set(add_obs or [])) - set(del_obs or [])
        for r, c in obs:
            grid[r, c] = OBSTACLE
        mov = (self.base_mov | set(add_mov or [])) - set(del_mov or [])
        return grid, list(mov), self.start, self.goal
 



def show_path_with_arrow(grid, path):
    start_pos = path[0][0]
    goal_pos = path[-1][0]
    movable_pos = path[-1][1]
    path_pos = [x[0] for x in path]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    pos_count_total = defaultdict(int)
    edge_count_total = defaultdict(int)
    for pos in path_pos:
        pos_count_total[pos] += 1
    for i in range(len(path_pos) - 1):
        edge = (path_pos[i], path_pos[i + 1])
        edge_count_total[edge] += 1

    pos_count_used = defaultdict(int)
    edge_count_used = defaultdict(int)

    cmap = cm.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=0, vmax=len(path_pos) - 2)

    legend_elements = []
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', label='Start',
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Goal',
               markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Movable',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='obstacle',
               markerfacecolor='grey', markersize=10)
    ]


    for i in range(len(path_pos) - 1):
        curr = path_pos[i]
        next_ = path_pos[i + 1]
        edge = (curr, next_)
        total = edge_count_total[edge]
        used = edge_count_used[edge]
        edge_count_used[edge] += 1

        cx, cy = curr[1], curr[0]
        dx = next_[1] - curr[1]
        dy = next_[0] - curr[0]

        if total == 1:
            offset_x, offset_y = 0, 0
        else:
            r = 0.1
            angle = 2 * math.pi * used / total
            offset_x = r * math.cos(angle + math.pi / 2)
            offset_y = r * math.sin(angle + math.pi / 2)

        arrow_color = cmap(norm(i))

        ax.arrow(cx + offset_x, cy + offset_y,
                 dx * 0.9, dy * 0.9,
                 head_width=0.1, head_length=0.1,
                 fc=arrow_color, ec=arrow_color, alpha=0.9,
                 length_includes_head=True)

        ax.scatter(cx, cy, color='orange', s=80, alpha=0.6)
        # 添加编号文字
        text_x = cx + offset_x + dx * 0.3
        text_y = cy + offset_y + dy * 0.3
        ax.text(text_x, text_y, str(i + 1), fontsize=7,
                color='black', ha='center', va='center', zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        
        legend_elements.append(Line2D([0], [0], color=arrow_color, lw=2, marker='>',
                                      label=f'Step {i + 1}', markersize=6))
        
    # ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0], color='orange', s=100, alpha=0.6)

    for m in movable_pos:
        ax.scatter(m[1], m[0], color='blue', s=100)

    ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=80, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='red', s=80, label='Goal')



    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
    # plt.title("Multicolor Path Arrows with Step Legend")
    plt.tight_layout()
    plt.show()

# save the figure to a file
def save_path_with_arrow(grid, path, output_file, dpi=150):
    start_pos = path[0][0]
    goal_pos = path[-1][0]
    movable_pos = path[-1][1]
    path_pos = [x[0] for x in path]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    pos_count_total = defaultdict(int)
    edge_count_total = defaultdict(int)
    for pos in path_pos:
        pos_count_total[pos] += 1
    for i in range(len(path_pos) - 1):
        edge = (path_pos[i], path_pos[i + 1])
        edge_count_total[edge] += 1

    pos_count_used = defaultdict(int)
    edge_count_used = defaultdict(int)

    cmap = cm.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=0, vmax=len(path_pos) - 2)

    legend_elements = []
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', label='Start',
               markerfacecolor='green', markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Goal',
               markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Movable',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='obstacle',
               markerfacecolor='grey', markersize=10)
    ]

    for i in range(len(path_pos) - 1):
        curr = path_pos[i]
        next_ = path_pos[i + 1]
        edge = (curr, next_)
        total = edge_count_total[edge]
        used = edge_count_used[edge]
        edge_count_used[edge] += 1

        cx, cy = curr[1], curr[0]
        dx = next_[1] - curr[1]
        dy = next_[0] - curr[0]

        if total == 1:
            offset_x, offset_y = 0, 0
        else:
            r = 0.1
            angle = 2 * math.pi * used / total
            offset_x = r * math.cos(angle + math.pi / 2)
            offset_y = r * math.sin(angle + math.pi / 2)

        arrow_color = cmap(norm(i))

        ax.arrow(cx + offset_x, cy + offset_y,
                 dx * 0.9, dy * 0.9,
                 head_width=0.1, head_length=0.1,
                 fc=arrow_color, ec=arrow_color, alpha=0.9,
                 length_includes_head=True)

        ax.scatter(cx, cy, color='orange', s=80, alpha=0.6)
        # 添加编号文字
        text_x = cx + offset_x + dx * 0.3
        text_y = cy + offset_y + dy * 0.3
        ax.text(text_x, text_y, str(i + 1), fontsize=7,
                color='black', ha='center', va='center', zorder=10,
                path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        
        legend_elements.append(Line2D([0], [0], color=arrow_color, lw=2, marker='>',
                                      label=f'Step {i + 1}', markersize=6))
        
    # ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=100, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0], color='orange', s=100, alpha=0.6)

    for m in movable_pos:
        ax.scatter(m[1], m[0], color='blue', s=100)

    ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=80, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='red', s=80, label='Goal')



    # ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)

    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)



def generate_steps_gif_with_arrows(grid, path, output_file, fps=3, dpi=150, max_steps=None):
    """
    生成逐步 GIF：
    - 用圆点累计显示到当前步的所有 step 位置；
    - 在相邻两步之间画箭头（带轻微偏移，避免多次相同边重叠）；
    - 显示 x/y 坐标轴与栅格（与 save_path_with_arrow 风格一致）。

    Parameters
    ----------
    grid : np.ndarray
    path : list[(pos, movable_set)]
        与现有代码一致的 path 结构（每个元素为 (位置tuple, movable_positions_frozenset)）
    output_file : str
        GIF 输出路径
    fps : int
        帧率
    dpi : int
        每帧保存分辨率
    max_steps : int or None
        只生成前 max_steps 步（用于截断演示）；默认 None 表示全部
    """
    import imageio
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import math
    import numpy as np

    if path is None or len(path) == 0:
        raise ValueError("Path is empty; cannot generate GIF.")

    # 取出关键元素
    rows, cols = grid.shape
    path_pos = [state[0] for state in path]
    start_pos = path_pos[0]
    goal_pos  = path_pos[-1]
    movable_pos_final = path[-1][1]  # 和你现有风格一致：展示最终的 movables

    # 统计边次数（用于重复边的偏移）
    edge_count_total = defaultdict(int)
    for i in range(len(path_pos) - 1):
        edge = (path_pos[i], path_pos[i + 1])
        edge_count_total[edge] += 1

    # 颜色映射：为每个“边序号”分配一个颜色
    cmap = cm.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=0, vmax=max(1, len(path_pos) - 2))

    frames = []
    # 逐帧绘制：第 i 帧展示到第 i 步（包含起点，i 从 0 开始）
    last_step = len(path_pos) - 1 if max_steps is None else min(max_steps, len(path_pos) - 1)
    for i in range(0, last_step + 1):
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

        # 背景栅格 + 轴：与 save_path_with_arrow 一致
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
        # 主刻度显示整数坐标
        ax.set_xticks(np.arange(0, cols, 2))
        ax.set_yticks(np.arange(0, rows, 2))
        # 次刻度用于画网格线（单元格边界在整数+0.5）
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

        # 画最终的 movable（与你现有风格保持一致）
        if movable_pos_final:
            ax.scatter([m[1] for m in movable_pos_final], [m[0] for m in movable_pos_final],
                       color=COLOR_MOVABLE, s=100)

        # 画起点/终点
        ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=80)
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=100)

        # ——累计画“圆点步骤”——
        # 0..i 的所有 step 用圆点（COLOR_AGENT_STEP）
        for j in range(0, i + 1):
            px, py = path_pos[j]
            ax.scatter(py, px, color=COLOR_AGENT_STEP, s=90, marker='o', alpha=0.85)

        # ——累计画“箭头”——（最多到 i-1）
        edge_count_used = defaultdict(int)
        for j in range(0, i):
            curr = path_pos[j]
            nxt  = path_pos[j + 1]
            edge = (curr, nxt)
            total = edge_count_total[edge]
            used  = edge_count_used[edge]
            edge_count_used[edge] += 1

            cx, cy = curr[1], curr[0]
            dx = nxt[1] - curr[1]
            dy = nxt[0] - curr[0]

            # 如果同一条边多次出现，给一点圆形偏移，避免完全重叠
            if total == 1:
                offset_x, offset_y = 0.0, 0.0
            else:
                r = 0.12
                angle = 2 * math.pi * used / total
                offset_x = r * math.cos(angle + math.pi / 2)
                offset_y = r * math.sin(angle + math.pi / 2)

            arrow_color = cmap(norm(j))
            ax.arrow(cx + offset_x, cy + offset_y,
                     dx * 0.9, dy * 0.9,
                     head_width=0.12, head_length=0.12,
                     fc=arrow_color, ec=arrow_color, alpha=0.95,
                     length_includes_head=True)

            # 在箭头中间放一个小编号，增强可读性（可选）
            text_x = cx + offset_x + dx * 0.5
            text_y = cy + offset_y + dy * 0.5
            ax.text(text_x, text_y, str(j + 1), fontsize=7, color='black',
                    ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        plt.tight_layout()

        # 转帧（复用你已有的 fig_to_image）
        frame = fig_to_image(fig)
        frames.append(frame)

    imageio.mimsave(output_file, frames, fps=fps)
    print(f"Steps-with-arrows GIF saved to: {output_file}")


def generate_steps_gif_with_arrows_2(grid, path, output_file, fps=3, dpi=150, max_steps=None):
    """
    逐步 GIF（圆点 + 箭头 + 坐标轴 + 每步自己的 movable_pos）
    - 每一帧：展示从起点到当前步的圆点轨迹；
    - 相邻两步之间画箭头（带轻微偏移，避免重叠）；
    - x/y 坐标轴与网格；
    - 关键变化：每一帧使用 path[i][1] 作为该帧的 movable_pos，直观看到每步对 movables 的更新。
    """
    import imageio
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.patheffects as pe
    import math
    import numpy as np

    if path is None or len(path) == 0:
        raise ValueError("Path is empty; cannot generate GIF.")

    rows, cols = grid.shape
    path_pos = [state[0] for state in path]
    start_pos = path_pos[0]
    goal_pos  = path_pos[-1]

    # 统计边出现次数（为重复边提供偏移）
    edge_count_total = defaultdict(int)
    for i in range(len(path_pos) - 1):
        edge = (path_pos[i], path_pos[i + 1])
        edge_count_total[edge] += 1

    cmap = cm.get_cmap('hsv')
    norm = mcolors.Normalize(vmin=0, vmax=max(1, len(path_pos) - 2))

    frames = []
    last_step = len(path_pos) - 1 if max_steps is None else min(max_steps, len(path_pos) - 1)

    for i in range(0, last_step + 1):
        # ——该帧的 movable 使用“当前步”的 movable_pos——
        current_movables = path[i][1]  # 可能是 list/tuple/set/frozenset
        # 统一成列表[(r,c), ...]
        if current_movables is None:
            current_movables_list = []
        else:
            current_movables_list = list(current_movables)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

        # 背景与坐标轴（与 save_path_with_arrow 风格一致）
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
        ax.set_xticks(np.arange(0, cols, 2))
        ax.set_yticks(np.arange(0, rows, 2))
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=True, left=True, labelbottom=True, labelleft=True)

        # 画当前帧对应的 movable（展示每一步的变化）
        if len(current_movables_list) > 0:
            ax.scatter([m[1] for m in current_movables_list],
                       [m[0] for m in current_movables_list],
                       color=COLOR_MOVABLE, s=100)

        # 起点/终点
        ax.scatter(start_pos[1], start_pos[0], marker='o', color=COLOR_AGENT, s=80)
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color=COLOR_GOAL, s=80)

        # 累计画“圆点步骤”：0..i
        for j in range(0, i + 1):
            px, py = path_pos[j]
            ax.scatter(py, px, color=COLOR_AGENT_STEP, s=80, marker='o', alpha=0.85)

        # 累计画“箭头”：0..i-1
        edge_count_used = defaultdict(int)
        for j in range(0, i):
            curr = path_pos[j]
            nxt  = path_pos[j + 1]
            edge = (curr, nxt)
            total = edge_count_total[edge]
            used  = edge_count_used[edge]
            edge_count_used[edge] += 1

            cx, cy = curr[1], curr[0]
            dx = nxt[1] - curr[1]
            dy = nxt[0] - curr[0]

            # 多次相同边的轻微偏移
            if total == 1:
                offset_x, offset_y = 0.0, 0.0
            else:
                r = 0.12
                angle = 2 * math.pi * used / total
                offset_x = r * math.cos(angle + math.pi / 2)
                offset_y = r * math.sin(angle + math.pi / 2)

            arrow_color = cmap(norm(j))
            ax.arrow(cx + offset_x, cy + offset_y,
                     dx * 0.9, dy * 0.9,
                     head_width=0.12, head_length=0.12,
                     fc=arrow_color, ec=arrow_color, alpha=0.95,
                     length_includes_head=True)

            # 箭头中点编号（可读性）
            text_x = cx + offset_x + dx * 0.5
            text_y = cy + offset_y + dy * 0.5
            ax.text(text_x, text_y, str(j + 1), fontsize=7, color='black',
                    ha='center', va='center',
                    path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        plt.tight_layout()
        frame = fig_to_image(fig)  # 复用你已有的工具
        frames.append(frame)

    imageio.mimsave(output_file, frames, fps=fps)
    print(f"Steps-with-arrows GIF saved to: {output_file}")