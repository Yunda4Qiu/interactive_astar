import os
import numpy as np
from matplotlib.patches import Rectangle
import imageio
import matplotlib.pyplot as plt
from config import GridcellValue

# 定义常量
OBSTACLE = GridcellValue.OBSTACLE.value
FREE     = GridcellValue.FREE.value


# ----------------------------------------
# 可视化与动画生成函数
# ----------------------------------------
def draw_grid(grid, agent, goal, movable, explored=None, path=None, title=""):
    """
    绘制 grid：
      - 黑色：障碍物
      - 白色：自由空间
      - 浅蓝色：探索过的节点（agent 位置）
      - 黄色：最终路径
      - 蓝色：可移动物体
      - 绿色：agent 当前所在
      - 红色：目标位置
    """
    rows, cols = grid.shape
    fig, ax = plt.subplots(figsize=(int(cols*0.4), int(rows*0.4)))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.invert_yaxis()  # (0,0)在左上角
    
    # 绘制每个单元格（自由空间/障碍物）
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == OBSTACLE:
                rect = Rectangle((j, i), 1, 1, facecolor='black')
            else:
                rect = Rectangle((j, i), 1, 1, facecolor='white')
            ax.add_patch(rect)
    
    # 绘制探索过的节点（仅绘制agent位置）
    if explored:
        for state in explored:
            pos, _ = state
            circle = plt.Circle((pos[1] + 0.5, pos[0] + 0.5), 0.2, color='lightblue', alpha=0.5)
            ax.add_patch(circle)
    
    # 绘制最终路径（黄色半透明覆盖）
    if path:
        path_positions = [state[0] for state in path]
        for pos in path_positions:
            rect = Rectangle((pos[1], pos[0]), 1, 1, facecolor='yellow', alpha=0.5)
            ax.add_patch(rect)
    
    # 绘制可移动物体（蓝色）
    for m in movable:
        circle = plt.Circle((m[1] + 0.5, m[0] + 0.5), 0.3, color='blue')
        ax.add_patch(circle)
    
    # 绘制 agent（绿色）
    circle = plt.Circle((agent[1] + 0.5, agent[0] + 0.5), 0.3, color='green')
    ax.add_patch(circle)
    
    # 绘制目标（红色）
    circle = plt.Circle((goal[1] + 0.5, goal[0] + 0.5), 0.3, color='red')
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
    # 根据图像尺寸和 dpi 计算真实像素宽高
    width, height = fig.get_size_inches() * fig.dpi
    width, height = int(width), int(height)
    # 使用 buffer_rgba 替代 tostring_rgb
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape((height, width, 4))
    image = image[:, :, :3]  # 取前三个通道，即 RGB
    plt.close(fig)
    return image


def generate_exploration_gif(grid, goal, explored_states, final_path, output_file):
    """
    生成一个 GIF，依次展示算法扩展的状态（agent及其周围状态）
    并在最后显示最终路径（最终状态中 agent 与可移动物体的位置）
    """
    frames = []
    num_frames = 20
    step_interval = max(1, len(explored_states) // num_frames)
    for i in range(0, len(explored_states), step_interval):
        explored_subset = explored_states[:i + 1]
        # 以当前扩展的最后状态作为绘图参考
        current_state = explored_states[i]
        agent, movable = current_state
        fig, ax = draw_grid(grid, agent, goal, movable, explored_subset, title=f"Explorate Step {i + 1}")
        frame = fig_to_image(fig)
        frames.append(frame)
    # 添加最终状态与路径显示
    final_state = final_path[-1]
    agent, movable = final_state
    fig, ax = draw_grid(grid, agent, goal, movable, explored_states, final_path, title="Final Path")
    frame = fig_to_image(fig)
    frames.append(frame)
    
    imageio.mimsave(output_file, frames, fps=3)
    print(f"Exploration GIF Save to: {output_file}")


def generate_steps_gif(grid, goal, path, output_file):
    """
    生成一个 GIF，逐步显示最终路径上每一步的状态变化，
    每一帧只显示从起点到当前 agent 所在位置的部分路径（黄色覆盖），
    而不是直接显示整个路径。
    """
    frames = []
    for idx, state in enumerate(path):
        agent, movable = state
        # 仅显示从起点到当前步的部分路径
        partial_path = path[:idx + 1]
        fig, ax = draw_grid(grid, agent, goal, movable, path=partial_path, title=f"Step: {idx + 1}")
        frame = fig_to_image(fig)
        frames.append(frame)
    imageio.mimsave(output_file, frames, fps=3)
    print(f"Steps GIF save to: {output_file}")



def show_grid(grid, start, goal, movable):
    """
    绘制 grid，显示起点、目标和所有可移动物体。
    
    参数:
      grid: 二维 numpy 数组，表示环境地图
      start: 起点坐标 (行, 列)
      goal: 目标坐标 (行, 列)
      movable: 可移动物体列表，每个元素为 (行, 列)
    """
    fig, ax = draw_grid(grid, start, goal, movable)
    # show the x and y sticks
    ax.set_xticklabels(np.arange(0, grid.shape[1] + 1))
    ax.set_yticklabels(np.arange(0, grid.shape[0] + 1))
    # ax.set_xticks(np.arange(0, grid.shape[1], 1))
    # ax.set_yticks(np.arange(0, grid.shape[0], 1))
    plt.show()

def save_grid(grid, start, goal, movable, output_file):
    """
    绘制整个 grid，显示起点、目标和所有可移动物体，并保存成图片。
    
    参数:
      grid: 二维 numpy 数组，表示环境地图
      start: 起点坐标 (行, 列)
      goal: 目标坐标 (行, 列)
      movable: 可移动物体列表，每个元素为 (行, 列)
      output_file: 保存的图片文件路径，例如 "grid.png"
    """
    # 这里把 agent 位置设为 start，用来显示起点位置
    fig, ax = draw_grid(grid, start, goal, movable, title="Initial Grid Setup")
    fig.savefig(output_file)
    plt.close(fig)
    print(f"Grid image saved to: {output_file}")

# Only the path length's cost, not include the PUSH_PENALTY
def calculate_path_cost(path):
    path_length = len(path)
    path_position = [x[0] for x in path]
    total_cost = 0

    for i in range(path_length - 1):
        pos1 = path_position[i]
        pos2 = path_position[i + 1]
        dx = abs(pos2[0] - pos1[0])
        dy = abs(pos2[1] - pos1[1])
        cost = np.sqrt(dx + dy)
        total_cost += cost
    return path_length, total_cost


def print_path_and_exploration_length(path, exploration):
    if path is not None:
        print(f"Path's length: {len(path)}")
        print(f"Exploration's length: {len(exploration)}")

def show_path(grid, path):
    start_pos = path[0][0]  # Start position
    goal_pos = path[-1][0]  # Goal position
    # original_movable_pos = path[0][1]
    movable_pos = path[-1][1]  # Last state's movable positions
    path_pos = [x[0] for x in path]  # Extract path positions

    fig, ax = plt.subplots(figsize=(6, 6))

    # Display the grid with transparency
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)  

    # Add grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    


    # Plot elements
    ax.scatter([p[1] for p in path_pos], [p[0] for p in path_pos], color='yellow', s=100, label='Path')
    ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color='blue', s=100, label='Movable')
    ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=80, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='red', s=80, label='Goal')

    # Show legend
    ax.legend()
    plt.show()

def save_path(grid, path, output_file, dpi=150):
    """
    Draws the grid + path + movable objects + start/goal
    and saves the plot to `output_file` (e.g. 'path.png').
    """
    start_pos = path[0][0]
    goal_pos = path[-1][0]
    movable_pos = path[-1][1]
    path_pos = [x[0] for x in path]

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background grid
    ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)

    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot path, movables, start, goal
    ax.scatter([p[1] for p in path_pos], [p[0] for p in path_pos],
               color='yellow', s=100, label='Path')
    ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos],
               color='blue', s=100, label='Movable')
    ax.scatter(start_pos[1], start_pos[0],
               marker='o', color='green', s=80, label='Start')
    ax.scatter(goal_pos[1], goal_pos[0],
               marker='*', color='red', s=80, label='Goal')

    ax.legend(loc='upper right')

    # Save and clean up
    fig.savefig(output_file, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def show_every_step(grid, path):
    """
    Show each step of the agent and movable objects in subplots.
    Each subplot shows a step in the path.

    Parameters:
    - grid: 2D numpy array of the environment.
    - path: List of tuples (position, movable_positions).
    """
    num_steps = len(path)

    # Determine subplot grid size (e.g., 4x4, 5x5) based on number of steps
    cols = min(num_steps, 5)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(-1)  # Flatten in case of 2D array

    for i, (pos, movable_pos) in enumerate(path):
        ax = axes[i]

        # Show the grid
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)

        # Grid lines
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

        # Hide ticks
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

        # Plot movable objects
        ax.scatter([m[1] for m in movable_pos], [m[0] for m in movable_pos], color='blue', s=60, label='Movable')

        # Plot current position of the agent
        ax.scatter(pos[1], pos[0], color='orange', s=80, marker='o', label='Agent')

        # Start and goal for reference
        start_pos = path[0][0]
        goal_pos = path[-1][0]
        ax.scatter(start_pos[1], start_pos[0], marker='o', color='green', s=10)
        ax.scatter(goal_pos[1], goal_pos[0], marker='*', color='red', s=60)

        ax.set_title(f"Step {i+1}")

    # Hide unused subplots if any
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


import os

def save_every_step(grid, path, output_file, max_cols=5, dpi=150):
    """
    Save all steps of the agent & movable objects in one figure with subplots.

    Parameters:
    - grid: 2D numpy array of the environment.
    - path: List of tuples (position, movable_positions).
    - output_file: filepath (including extension) to save the combined figure.
    - max_cols: maximum number of columns in the subplot grid.
    - dpi: resolution of saved figure.
    """
    num_steps = len(path)
    cols = min(num_steps, max_cols)
    rows = (num_steps + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.array(axes).reshape(-1)  # flatten in case of 2D array

    start_pos = path[0][0]
    goal_pos  = path[-1][0]

    for i, (pos, movable_pos) in enumerate(path):
        ax = axes[i]
        ax.imshow(grid, cmap='gray_r', origin='upper', alpha=0.8)
        ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

        # Movable objects
        ax.scatter([m[1] for m in movable_pos],
                   [m[0] for m in movable_pos],
                   color='blue', s=60, label='Movable')
        # Agent
        ax.scatter(pos[1], pos[0],
                   color='orange', s=80, marker='o', label='Agent')
        # Start & goal
        ax.scatter(start_pos[1], start_pos[0],
                   marker='o', color='green', s=10)
        ax.scatter(goal_pos[1], goal_pos[0],
                   marker='*', color='red', s=60)

        ax.set_title(f"Step {i+1}")

    # Turn off any unused axes
    for j in range(num_steps, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    # Save the combined figure
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
        # 初始化
        grid = np.zeros((self.rows, self.cols), dtype=int)
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OBSTACLE
        # 处理障碍
        obs = (self.base_obs | set(add_obs or [])) - set(del_obs or [])
        for r,c in obs:
            grid[r,c] = OBSTACLE
        # 处理可动物体
        mov = (self.base_mov | set(add_mov or [])) - set(del_mov or [])
        return grid, list(mov), self.start, self.goal
