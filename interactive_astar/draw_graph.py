import matplotlib.pyplot as plt
import matplotlib.patches as patches

GRID_N = 4  # 4x4 map

ROBOT_COLOR = (0.0, 0.6, 0.2)      # green
ROBOT_TARGET_ALPHA = 0.35
MOVABLE_COLOR = (0.2, 0.45, 0.75)  # blue
MOVABLE_TARGET_ALPHA = 0.28

def draw_grid(ax, n=GRID_N):
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')
    for i in range(n+1):
        ax.plot([0, n], [i, i], color='lightgray', linewidth=0.8, zorder=0)
        ax.plot([i, i], [0, n], color='lightgray', linewidth=0.8, zorder=0)

def draw_robot(ax, pos, alpha=1.0, z=5):
    ax.add_patch(patches.Circle((pos[0]+0.5, pos[1]+0.5), 0.33,
                                facecolor=ROBOT_COLOR, edgecolor='black',
                                linewidth=1.0, alpha=alpha, zorder=z))

def draw_movable(ax, pos, alpha=0.9, z=3):
    ax.add_patch(patches.Rectangle((pos[0], pos[1]), 1, 1,
                                   facecolor=MOVABLE_COLOR, edgecolor='black',
                                   linewidth=1.0, alpha=alpha, zorder=z))

def draw_movable_target(ax, pos, alpha=MOVABLE_TARGET_ALPHA, z=2):
    rect = patches.Rectangle((pos[0], pos[1]), 1, 1,
                             facecolor=MOVABLE_COLOR, edgecolor=MOVABLE_COLOR,
                             linewidth=1.2, linestyle='--', hatch='///',
                             alpha=alpha, zorder=z)
    ax.add_patch(rect)

def draw_arrow(ax, src, d, z=6):
    ax.arrow(src[0]+0.5, src[1]+0.5, d[0], d[1],
             head_width=0.22, head_length=0.22, fc='black', ec='black', zorder=z)

def draw_action(ax, action_type):
    draw_grid(ax, GRID_N)

    # default placement for 4x4
    robot_pos = (1, 1)

    if action_type == "Normal":
        draw_robot(ax, robot_pos, alpha=1.0, z=6)
        draw_robot(ax, (2, 1), alpha=ROBOT_TARGET_ALPHA, z=7)
        draw_arrow(ax, robot_pos, (1, 0))

    elif action_type == "Push":
        movable_pos = (2, 1)
        draw_robot(ax, robot_pos, alpha=1.0, z=6)
        draw_robot(ax, movable_pos, alpha=ROBOT_TARGET_ALPHA, z=7)     # robot target on top
        draw_movable(ax, movable_pos, alpha=0.9, z=4)
        draw_movable_target(ax, (3, 1), alpha=MOVABLE_TARGET_ALPHA, z=3)
        draw_arrow(ax, robot_pos, (1, 0))

    elif action_type == "Pull":
        movable_pos = (2, 1)
        retreat_pos = (0, 1)
        draw_robot(ax, robot_pos, alpha=1.0, z=6)
        draw_robot(ax, retreat_pos, alpha=ROBOT_TARGET_ALPHA, z=7)     # robot target on top
        draw_movable(ax, movable_pos, alpha=0.9, z=4)
        draw_movable_target(ax, (1, 1), alpha=MOVABLE_TARGET_ALPHA, z=3)  # object -> robot's old cell
        draw_arrow(ax, robot_pos, (-1, 0))

    elif action_type == "Double Push":
        # two movables in a line; shift as a unit by one cell right
        first_pos  = (1, 1)   # closest to robot
        second_pos = (2, 1)
        robot_start = (0, 1)
        draw_robot(ax, robot_start, alpha=1.0, z=6)
        draw_robot(ax, first_pos, alpha=ROBOT_TARGET_ALPHA, z=7)       # robot target at first_pos
        draw_movable(ax, first_pos, alpha=0.9, z=4)
        draw_movable(ax, second_pos, alpha=0.9, z=4)
        draw_movable_target(ax, second_pos, alpha=MOVABLE_TARGET_ALPHA, z=3)  # first -> second's old cell
        draw_movable_target(ax, (3, 1), alpha=MOVABLE_TARGET_ALPHA, z=3)      # second -> new free cell
        draw_arrow(ax, robot_start, (1, 0))

    elif action_type == "Diagonal Push":
        robot_pos = (1, 1)
        movable_pos = (2, 2)
        draw_robot(ax, robot_pos, alpha=1.0, z=6)
        draw_robot(ax, movable_pos, alpha=ROBOT_TARGET_ALPHA, z=7)     # robot target on top
        draw_movable(ax, movable_pos, alpha=0.9, z=4)
        draw_movable_target(ax, (3, 3), alpha=MOVABLE_TARGET_ALPHA, z=3)
        draw_arrow(ax, robot_pos, (1, 1))

    elif action_type == "Diagonal Pull":
        robot_pos = (1, 1)
        movable_pos = (2, 2)
        retreat_pos = (0, 0)
        draw_robot(ax, robot_pos, alpha=1.0, z=6)
        draw_robot(ax, retreat_pos, alpha=ROBOT_TARGET_ALPHA, z=7)     # robot target on top
        draw_movable(ax, movable_pos, alpha=0.9, z=4)
        draw_movable_target(ax, (1, 1), alpha=MOVABLE_TARGET_ALPHA, z=3)
        draw_arrow(ax, robot_pos, (-1, -1))

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
actions = ["Normal", "Push", "Pull", "Double Push", "Diagonal Push", "Diagonal Pull"]
for ax, act in zip(axes.flat, actions):
    draw_action(ax, act)
    # 在底部添加标题
    ax.text(0.5, -0.01, act, transform=ax.transAxes,
            ha='center', va='top', fontsize=11)
plt.tight_layout()
plt.savefig("interactive_astar_actions.png", dpi=300, bbox_inches='tight')
# plt.savefig("interactive_astar_action_types.pdf", bbox_inches='tight')
plt.show()