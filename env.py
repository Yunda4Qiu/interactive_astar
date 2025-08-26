from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ------------------------------
# CONSTANTS
# ------------------------------
FREE = 0
OBSTACLE = 1
PUSH_PENALTY = 0.3  # 仅用于微步内部代价选择（非环境 reward），可按需调整

# 8 邻动作（坐标原点左上：X 向下为正，Y 向右为正）
MOVES_8: List[Tuple[int, int]] = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1)
]
# 正交四方向（保持为 list，顺序稳定）
CARDINAL_4: List[Tuple[int, int]] = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def move_cost(delta: Tuple[int, int]) -> float:
    return 1.0 if abs(delta[0]) + abs(delta[1]) == 1 else math.sqrt(2.0)


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2.0) - 2.0) * min(dx, dy)


# ------------------------------
# 配置
# ------------------------------
@dataclass
class GridConf:
    H: int = 10
    W: int = 10
    gap_block_prob: float = 0.8          # 缺口处放 movable 的概率
    gap_distance_prob1: float = 0.8      # 两线间距=1 的概率
    max_steps: int = 200
    seed: int = 0


class PushPullGridEnv:
    metadata = {"render_modes": []}

    def __init__(self, conf: GridConf):
        self.conf = conf
        self.rng = random.Random(conf.seed)
        self.np_rng = np.random.default_rng(conf.seed)

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4, conf.H, conf.W), dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.grid_static = np.zeros((conf.H, conf.W), dtype=np.int32)
        self.movable: List[Tuple[int, int]] = []
        self.agent: Tuple[int, int] = (1, 1)
        self.goal: Tuple[int, int] = (conf.H - 2, conf.W - 2)
        self.steps = 0

        # 轨迹与动作记录
        self.path: List[Tuple[int, int]] = []
        self.action_hist: List[int] = []

        self.reset(seed=conf.seed)

    # ------------------------------
    # 地图生成（严格按规则）
    # ------------------------------
    def _generate_map(self):
        H, W = self.conf.H, self.conf.W
        g = np.zeros((H, W), dtype=np.int32)
        # 边框障碍
        g[0, :] = OBSTACLE
        g[H - 1, :] = OBSTACLE
        g[:, 0] = OBSTACLE
        g[:, W - 1] = OBSTACLE

        self.agent = (1, 1)
        # self.goal = (H - 2, W - 2)
        self.goal = (H-2, W - 2)  # 目标在右上角，便于测试

        def draw_vline_with_one_gap(col: int, gap_row: int, r0: int = 1, r1: int | None = None):
            if r1 is None:
                r1 = H - 2
            for r in range(r0, r1 + 1):
                if r == gap_row:
                    continue
                g[r, col] = OBSTACLE

        def draw_hline_with_one_gap(row: int, gap_col: int, c0: int = 1, c1: int | None = None):
            if c1 is None:
                c1 = W - 2
            for c in range(c0, c1 + 1):
                if c == gap_col:
                    continue
                g[row, c] = OBSTACLE

        # 随机决定主线方向（水平 or 竖直）
        main_is_horizontal = (self.rng.random() < 0.5)
        # 两线间距：概率 -> 1，否则 2
        gap_dist = 1 if self.rng.random() < self.conf.gap_distance_prob1 else 2

        path_cells: List[Tuple[int, int]] = []      # 路径单元格（用于额外放 movable）
        movable_candidates: List[Tuple[int, int]] = []  # 缺口处候选 movable

        if main_is_horizontal:
            mid = H // 2
            r1 = self.rng.randrange(max(2, mid - 3), min(H - 3 - gap_dist - 1, mid + 2))
            r2 = r1 + gap_dist + 1

            gap_c1 = self.rng.randrange(1, W - 2)
            gap_c2 = self.rng.randrange(1, W - 2)
            draw_hline_with_one_gap(r1, gap_c1)
            draw_hline_with_one_gap(r2, gap_c2)
            movable_candidates += [(r1, gap_c1), (r2, gap_c2)]
            for c in range(1, W - 1):
                for rr in range(min(r1, r2) + 1, max(r1, r2)):
                    path_cells.append((rr, c))
        else:
            mid = W // 2
            c1 = self.rng.randrange(max(2, mid - 2), min(W - 3 - gap_dist - 1, mid + 2))
            c2 = c1 + gap_dist + 1
            gap_r1 = self.rng.randrange(1, H - 2)
            gap_r2 = self.rng.randrange(1, H - 2)
            draw_vline_with_one_gap(c1, gap_r1)
            draw_vline_with_one_gap(c2, gap_r2)
            movable_candidates += [(gap_r1, c1), (gap_r2, c2)]
            for r in range(1, H - 1):
                for cc in range(min(c1, c2) + 1, max(c1, c2)):
                    path_cells.append((r, cc))

        # 起终点设为可走
        g[self.agent] = FREE
        g[self.goal] = FREE

        # 连通性：若不通则打孔修复
        if not self._is_connected(g, self.agent, self.goal):
            self._punch_for_connectivity(g, self.agent, self.goal)

        self.grid_static = g

        # 缺口处放 movable
        self.movable = []
        for p in movable_candidates:
            if p in (self.agent, self.goal):
                continue
            if self._inside_and_free_grid(g, p) and (self.rng.random() < self.conf.gap_block_prob):
                self.movable.append(p)
        # 增加 1~2 个通道内 movable，提升交互概率
        if main_is_horizontal:
            extras = 1 if abs(r1 - r2) == 2 else 2
        else:
            extras = 1 if abs(c1 - c2) == 2 else 2
        self.np_rng.shuffle(path_cells)
        for p in path_cells:
            if len(self.movable) >= len(movable_candidates) + extras:
                break
            if p in (self.agent, self.goal) or (p in self.movable):
                continue
            if self._inside_and_free_grid(g, p):
                self.movable.append(p)

    # ------------------------------
    # 连通性 & 打孔
    # ------------------------------
    def _is_connected(self, g: np.ndarray, s: Tuple[int, int], t: Tuple[int, int]) -> bool:
        H, W = g.shape
        from collections import deque
        vis = np.zeros_like(g, dtype=bool)
        dq = deque([s])
        vis[s] = True
        while dq:
            x, y = dq.popleft()
            if (x, y) == t:
                return True
            for dx, dy in MOVES_8:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W and (not vis[nx, ny]) and (g[nx, ny] != OBSTACLE):
                    vis[nx, ny] = True
                    dq.append((nx, ny))
        return False

    def _punch_for_connectivity(self, g: np.ndarray, s: Tuple[int, int], t: Tuple[int, int]):
        H, W = g.shape
        for _ in range(64):
            if self._is_connected(g, s, t):
                break
            r = self.rng.randrange(1, H - 1)
            c = self.rng.randrange(1, W - 1)
            if g[r, c] == OBSTACLE:
                g[r, c] = FREE

    # ------------------------------
    # 基础合法性
    # ------------------------------
    def _inside_and_free_grid(self, g: np.ndarray, p: Tuple[int, int]) -> bool:
        H, W = g.shape
        x, y = p
        return 0 <= x < H and 0 <= y < W and g[x, y] != OBSTACLE

    def _inside_and_free(self, p: Tuple[int, int]) -> bool:
        return self._inside_and_free_grid(self.grid_static, p)

    def _is_movable(self, p: Tuple[int, int]) -> bool:
        return p in self.movable

    # 返回把 obj（movable）推向其四邻的所有可用目的地（不含 agent 当前格）
    def _push_candidates(self, obj: Tuple[int, int], agent: Tuple[int, int], movtuple: Optional[Tuple[Tuple[int,int], ...]] = None) -> List[Tuple[int, int]]:
        cands: List[Tuple[int, int]] = []
        movset = set(movtuple) if movtuple is not None else set(self.movable)
        for dx, dy in CARDINAL_4:
            q = (obj[0] + dx, obj[1] + dy)
            if not self._inside_and_free(q):
                continue
            if q in movset:
                continue
            if q == agent:
                continue
            cands.append(q)
        return cands

    # agent 与 obj 是否正交相邻（用于 pull 的先决）
    def _pull_possible(self, obj: Tuple[int, int], agent: Tuple[int, int]) -> bool:
        dx = agent[0] - obj[0]
        dy = agent[1] - obj[1]
        return (dx, dy) in CARDINAL_4 and self._inside_and_free(agent) and (agent not in self.movable)

    # 返回 agent 8 邻中可后退的格（不能是 exclude=movable 原位，不能踩其他 movable）
    def _retreat_candidates(self, agent: Tuple[int, int], exclude: Tuple[int, int], movtuple: Optional[Tuple[Tuple[int,int], ...]] = None) -> List[Tuple[int, int]]:
        cands: List[Tuple[int, int]] = []
        movset = set(movtuple) if movtuple is not None else set(self.movable)
        for dx, dy in MOVES_8:
            r = (agent[0] + dx, agent[1] + dy)
            if not self._inside_and_free(r):
                continue
            if r == exclude:
                continue
            if r in movset:
                continue
            cands.append(r)
        return cands

    # 新版动作合法性判定：完全符合三条规则
    def _direction_legal(self, delta: Tuple[int, int]) -> bool:
        dx, dy = delta
        ax, ay = self.agent
        dest = (ax + dx, ay + dy)

        # 出界或静态障碍
        if not (0 <= dest[0] < self.conf.H and 0 <= dest[1] < self.conf.W):
            return False
        if self.grid_static[dest] == OBSTACLE:
            return False

        # 目标格无 movable：普通移动允许（8 方向）
        if dest not in self.movable:
            return True

        # 目标格是 movable：
        push_ok = len(self._push_candidates(dest, self.agent)) > 0
        pull_ok = (dx, dy) in CARDINAL_4 and (len(self._retreat_candidates(self.agent, exclude=dest)) > 0)
        return push_ok or pull_ok

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(8, dtype=np.int8)
        for a, d in enumerate(MOVES_8):
            mask[a] = 1 if self._direction_legal(d) else 0
        return mask

    # ------------------------------
    # 纯模拟微步（供先验 lookahead 使用；不修改 self）
    # 返回：(next_agent, next_movables, cost, mode) 或 None
    # ------------------------------
    def _micro_placement_score(self, agent_after, mov_after, moved_from=None):
        reach = self._reachable_with_movables(agent_after, self.goal, mov_after)
        penalty = 0.0
        if not reach:
            penalty += 10.0  # pushing into a spot that blocks reachability
        if moved_from is not None and self._is_bottleneck_cell(moved_from) and moved_from in mov_after:
            penalty += 1.0   # avoid keeping a bottleneck blocked
        return penalty + octile_distance(agent_after, self.goal)


    def simulate_micro_step(self, agent: Tuple[int, int], movables: Tuple[Tuple[int,int], ...], delta: Tuple[int,int]) \
            -> Optional[Tuple[Tuple[int,int], List[Tuple[int,int]], float, str]]:
        ax, ay = agent
        dest = (ax + delta[0], ay + delta[1])

        if not self._inside_and_free(dest):
            return None

        if dest not in movables:  # 普通移动
            c = move_cost(delta)
            h = octile_distance(dest, self.goal)
            return dest, list(movables), c + h, "move"

        # dest 是 movable：尝试 push/pull
        best_cost = float('inf')
        best_pack = None

        # PUSH：动作可为 8 向
        push_cands = self._push_candidates(dest, agent, movtuple=movables)
        for q in push_cands:
            new_mov = list(movables)
            new_mov.remove(dest)
            new_mov.append(q)
            c = move_cost(delta) + PUSH_PENALTY * 1.0
            # h = octile_distance(dest, self.goal)  # agent 站到 dest
            # total = c + h
            total = c + self._micro_placement_score(dest, new_mov, moved_from=dest)
            if total < best_cost:
                best_cost = total
                best_pack = (dest, new_mov, total, "push")

        # PULL：仅正交；agent 退到 8 邻
        if delta in CARDINAL_4 and self._pull_possible(dest, agent):
            rets = self._retreat_candidates(agent, exclude=dest, movtuple=movables)
            for r in rets:
                new_mov = list(movables)
                new_mov.remove(dest)
                new_mov.append(agent)
                step_vec = (r[0] - agent[0], r[1] - agent[1])
                c = move_cost(step_vec) + PUSH_PENALTY * 1.0
                # h = octile_distance(r, self.goal)
                # total = c + h
                total = c + self._micro_placement_score(dest, new_mov, moved_from=dest)
                if total < best_cost:
                    best_cost = total
                    best_pack = (r, new_mov, total, "pull")

        return best_pack

    # ------------------------------
    # 与合法性判定一致的“贪心微步”（实际执行）
    # ------------------------------
    def _greedy_micro_step(self, delta: Tuple[int, int]) -> Optional[Tuple[Tuple[int, int], List[Tuple[int, int]], float, str]]:
        return self.simulate_micro_step(self.agent, tuple(sorted(self.movable)), delta)

    # ------------------------------
    # Reachability（movables 当作障碍）
    # ------------------------------
    def _reachable_with_movables(self, start: Tuple[int,int], goal: Tuple[int,int], movables: Optional[List[Tuple[int,int]]] = None) -> bool:
        if movables is None:
            movables = self.movable
        H, W = self.conf.H, self.conf.W
        from collections import deque
        blocked = set(movables)
        if start in blocked or goal in blocked:
            return False
        vis = np.zeros((H,W), dtype=bool)
        dq = deque([start])
        vis[start] = True
        while dq:
            x, y = dq.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in MOVES_8:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < H and 0 <= ny < W):
                    continue
                if vis[nx, ny]:
                    continue
                if self.grid_static[nx, ny] == OBSTACLE:
                    continue
                if (nx, ny) in blocked:
                    continue
                vis[nx, ny] = True
                dq.append((nx, ny))
        return False

    def _is_bottleneck_cell(self, p: Tuple[int,int]) -> bool:
        x, y = p
        # 越界或静态障碍位置不算
        if not (0 <= x < self.conf.H and 0 <= y < self.conf.W):
            return False
        if self.grid_static[x, y] == OBSTACLE:
            return False
        # 四邻
        def safe_ob(x,y):
            if not (0 <= x < self.conf.H and 0 <= y < self.conf.W):
                return True  # 边界外视为障碍，增强瓶颈感
            return self.grid_static[x, y] == OBSTACLE
        up = safe_ob(x-1, y)
        down = safe_ob(x+1, y)
        left = safe_ob(x, y-1)
        right = safe_ob(x, y+1)
        # 垂直走廊：左右都是障碍，上下为通路
        vertical_gate = left and right and (not up) and (not down)
        # 水平走廊：上下都是障碍，左右为通路
        horizontal_gate = up and down and (not left) and (not right)
        return vertical_gate or horizontal_gate

    # ------------------------------
    # 展示 & 可视化
    # ------------------------------
    def show_grid(self):
        H, W = self.conf.H, self.conf.W
        grid = np.full((H, W), ' ')
        for r in range(H):
            for c in range(W):
                if self.grid_static[r, c] == OBSTACLE:
                    grid[r, c] = '#'
                elif (r, c) in self.movable:
                    grid[r, c] = 'M'
                elif (r, c) == self.agent:
                    grid[r, c] = 'A'
                elif (r, c) == self.goal:
                    grid[r, c] = 'G'
        print("\n".join("".join(row) for row in grid))

    def render_path(self, path: Optional[List[Tuple[int, int]]] = None, 
                    movable: Optional[List[Tuple[int, int]]] = None, 
                    save_path: Optional[str] = None, 
                    add_legend: bool = False):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if path is None:
            path = self.path

        H, W = self.conf.H, self.conf.W
        if add_legend:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(-0.5, H - 0.5)

        # 设置次刻度：每格都画网格线
        ax.set_xticks(np.arange(0.5, W-0.5), minor=True)
        ax.set_yticks(np.arange(0.5, H-0.5), minor=True)

        # 网格线：主格粗一点，次格细一点
        ax.grid(which="minor", color="gray", linewidth=0.8)
        


        # 静态障碍
        for r in range(H):
            for c in range(W):
                if self.grid_static[r, c] == OBSTACLE:
                    rect = patches.Rectangle((c - 0.5, r - 0.5), 1, 1, linewidth=0.5, edgecolor='black', facecolor='darkgray', label='Obstacle')
                    ax.add_patch(rect)

        # movables（当前状态）
        if movable is None:
            movable = self.movable
        for m in movable:
            ax.plot(m[1], m[0], 'o', color='blue', markersize=8, label='Movable')

        # 起点/终点
        if self.path:
            ax.plot(self.path[0][1], self.path[0][0], 'o', color='green', markersize=12, label='Start')  # 起点
        ax.plot(self.goal[1], self.goal[0], '*', color='red', markersize=12, label='Goal')           # 终点

        # 路径（折线）
        if len(path) >= 2:
            xs = [p[1] for p in path]
            ys = [p[0] for p in path]
            ax.plot(xs, ys, '-', linewidth=2, color='orange', label='Path')
            ax.plot(xs, ys, 'o', markersize=8, color='orange', label='Path', alpha=0.5)  # 路径点

        if add_legend:
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            # 把legend放在图外面
            ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.01, 0.5), borderaxespad=0.)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()


    # ------------------------------
    # 传统 A*：8 向移动；静态障碍 + movables 都视为障碍
    # 返回: (path: List[Tuple[int,int]], total_cost: float)
    # ------------------------------
    def astar(self,
              start: Optional[Tuple[int, int]] = None,
              goal: Optional[Tuple[int, int]] = None) -> Tuple[List[Tuple[int, int]], float]:
        from heapq import heappush, heappop

        H, W = self.conf.H, self.conf.W
        start = self.agent if start is None else start
        goal = self.goal if goal is None else goal

        # 边界与可通行判断（movables 也当作障碍）
        blocked = set(self.movable)

        def inside_and_free(p: Tuple[int, int]) -> bool:
            x, y = p
            if not (0 <= x < H and 0 <= y < W):
                return False
            if self.grid_static[x, y] == OBSTACLE:
                return False
            if (x, y) in blocked:
                return False
            return True

        # 起止点若被占/越界，直接失败
        if (not inside_and_free(start)) or (not inside_and_free(goal)):
            return [], float('inf')

        # A* 主体
        def h(p: Tuple[int, int]) -> float:
            return octile_distance(p, goal)

        open_heap = []  # (f, g, node)
        g_best: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        heappush(open_heap, (h(start), 0.0, start))

        found = None

        while open_heap:
            f, g, cur = heappop(open_heap)
            # 已有更优 g，跳过
            if g > g_best.get(cur, float('inf')) + 1e-12:
                continue
            if cur == goal:
                found = cur
                break

            cx, cy = cur
            for dx, dy in MOVES_8:
                nxt = (cx + dx, cy + dy)
                if not inside_and_free(nxt):
                    continue

                step_c = move_cost((dx, dy))
                new_g = g + step_c

                if new_g + 1e-12 < g_best.get(nxt, float('inf')):
                    g_best[nxt] = new_g
                    parent[nxt] = cur
                    heappush(open_heap, (new_g + h(nxt), new_g, nxt))

        if found is None:
            return [], float('inf')

        # 回溯路径
        path: List[Tuple[int, int]] = []
        cur = found
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        path.reverse()

        total_cost = g_best[found]
        return path, total_cost
    

    # ------------------------------
    # A* interactive
    # ------------------------------
    def astar_interactive(self, return_movables: bool = False):
        from heapq import heappush, heappop

        H, W = self.conf.H, self.conf.W
        start_agent = self.agent
        start_mov = tuple(sorted(self.movable))  # 作为状态的一部分（可哈希）

        def inside_and_free_grid(p):
            x, y = p
            return 0 <= x < H and 0 <= y < W and self.grid_static[x, y] != OBSTACLE

        def is_movable_at(p, movtuple):
            return p in movtuple

        def push_candidates(obj, agent_pos, movtuple):
            out = []
            for dx, dy in CARDINAL_4:
                q = (obj[0] + dx, obj[1] + dy)
                if not inside_and_free_grid(q):
                    continue
                if q == agent_pos:
                    continue
                if q in movtuple:
                    continue
                out.append(q)
            return out

        def retreat_candidates(agent_pos, exclude, movtuple):
            out = []
            for dx, dy in MOVES_8:
                r = (agent_pos[0] + dx, agent_pos[1] + dy)
                if not inside_and_free_grid(r):
                    continue
                if r == exclude:
                    continue
                if r in movtuple:
                    continue
                out.append(r)
            return out

        def heuristic(agent_pos):
            return octile_distance(agent_pos, self.goal)

        def successors(agent_pos, movtuple):
            succs = []
            ax, ay = agent_pos
            for dx, dy in MOVES_8:
                dest = (ax + dx, ay + dy)
                if not inside_and_free_grid(dest):
                    continue

                if not is_movable_at(dest, movtuple):
                    step_c = move_cost((dx, dy))
                    succs.append((dest, movtuple, step_c, "move", (dx, dy), None))
                else:
                    obj = dest
                    for q in push_candidates(obj, agent_pos, movtuple):
                        new_mov = list(movtuple)
                        new_mov.remove(obj)
                        new_mov.append(q)
                        new_movtuple = tuple(sorted(new_mov))
                        step_c = move_cost((dx, dy)) + PUSH_PENALTY * 1.0
                        succs.append((obj, new_movtuple, step_c, "push", (dx, dy), {"pushed_to": q}))

                    if (dx, dy) in CARDINAL_4:
                        rets = retreat_candidates(agent_pos, exclude=obj, movtuple=movtuple)
                        if rets:
                            for r in rets:
                                new_mov = list(movtuple)
                                new_mov.remove(obj)
                                new_mov.append(agent_pos)
                                new_movtuple = tuple(sorted(new_mov))
                                step_vec = (r[0] - ax, r[1] - ay)
                                step_c = move_cost(step_vec) + PUSH_PENALTY * 1.0
                                succs.append((r, new_movtuple, step_c, "pull", (dx, dy), {"retreat_to": r}))
            return succs

        start_state = (start_agent, start_mov)
        goal_pos = self.goal

        open_heap = []
        g_best: Dict = {start_state: 0.0}
        heappush(open_heap, (heuristic(start_agent), 0.0, start_state))
        parent = {start_state: None}
        found_state = None

        while open_heap:
            f, g, state = heappop(open_heap)
            agent_pos, movtuple = state
            if g > g_best[state] + 1e-12:
                continue
            if agent_pos == goal_pos:
                found_state = state
                break
            for na, nmov, step_c, mode, via_delta, _extra in successors(agent_pos, movtuple):
                new_state = (na, nmov)
                new_g = g + step_c
                if (new_state not in g_best) or (new_g + 1e-12 < g_best[new_state]):
                    g_best[new_state] = new_g
                    parent[new_state] = (state, mode, via_delta)
                    heappush(open_heap, (new_g + heuristic(na), new_g, new_state))

        if found_state is None:
            return [], None if not return_movables else [], float('inf')

        path = []
        mov_hist = []
        cur = found_state
        while cur is not None:
            a_pos, mtuple = cur
            path.append(a_pos)
            mov_hist.append(list(mtuple))
            prev_pack = parent[cur]
            if prev_pack is None:
                break
            cur = prev_pack[0]

        path.reverse()
        mov_hist.reverse()

        total_cost = g_best[found_state]
        if return_movables:
            return path, mov_hist, total_cost
        else:
            return path, mov_hist[-1], total_cost

    # ------------------------------
    # Adapt to Gym API
    # ------------------------------
    def _build_obs(self) -> np.ndarray:
        H, W = self.conf.H, self.conf.W
        c0 = (self.grid_static == OBSTACLE).astype(np.float32)
        c1 = np.zeros((H, W), dtype=np.float32)
        for p in self.movable:
            c1[p] = 1.0
        c2 = np.zeros((H, W), dtype=np.float32); c2[self.agent] = 1.0
        c3 = np.zeros((H, W), dtype=np.float32); c3[self.goal] = 1.0
        return np.stack([c0, c1, c2, c3], axis=0)

    def normalize_reward(self, r, min_r=-20.0, max_r=40.0):
        return 2 * (r - min_r) / (max_r - min_r) - 1  # 映射到 [-1, 1]

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)
        self._generate_map()
        self.steps = 0
        # 记录初始化
        self.path = [self.agent]
        self.action_hist = []
        obs = self._build_obs()
        info = {"action_mask": self._action_mask()}
        return obs, info

    def step(self, action: int):
        self.steps += 1
        delta = MOVES_8[action]
        mask = self._action_mask()
        terminated = False
        truncated = False

        # ---------------- Reward constants ----------------
        R_GOAL         = 100.0   # 成功固定奖励（主奖励）
        R_FAIL         = -10.0   # 超时等失败的固定惩罚
        STEP_COST      = -1.0    # 每步时间成本（让回报与效率挂钩）
        ILLEGAL_COST   = -2.0    # 非法动作的额外惩罚（mask 正常时很少触发）
        INTERACT_COST  = -0.2    # push/pull 的小成本（保留，但不阻止清障）
        CONNECT_BONUS  =  +2.0   # “首次从不可达→可达”的小奖励（鼓励清障）

        # 势能 shaping（Ng+99）：F = α(γ φ(s') - φ(s))，不改变最优策略
        GAMMA_SHAPING  = 0.99
        ALPHA_SHAPING  = 0.20    # 小权重，避免中间奖励盖过终点奖励

        # 单步奖励下界，避免过大负值（曲线更稳）
        REWARD_MIN     = -20.0

        # ---------------- 1) 非法动作（被 mask 掉仍选择时） ----------------
        reward = STEP_COST
        if mask[action] == 0:
            reward += ILLEGAL_COST
            if self.steps >= self.conf.max_steps:
                truncated = True
                reward += R_FAIL
            reward = max(reward, REWARD_MIN)
            # reward = self.normalize_reward(reward)
            obs = self._build_obs()
            info = {"action_mask": mask}
            return obs, reward, terminated, truncated, info

        # ---------------- 2) 动作执行 ----------------
        prev_agent = self.agent
        phi_s = -octile_distance(prev_agent, self.goal)  # φ(s)

        before_reach = self._reachable_with_movables(self.agent, self.goal, self.movable)

        nxt = self._greedy_micro_step(delta) 
        if nxt is None:                       
            reward += ILLEGAL_COST
            if self.steps >= self.conf.max_steps:
                truncated = True
                reward += R_FAIL
            reward = max(reward, REWARD_MIN)
            
            obs = self._build_obs()
            info = {"action_mask": mask}
            return obs, reward, terminated, truncated, info

        next_agent, next_movables, _cost_unused, mode = nxt

        # 交互小成本（保留“有代价”的直觉，但不阻碍清障）
        if mode in ("push", "pull"):
            reward += INTERACT_COST

        # 应用状态
        self.agent = next_agent
        self.movable = list(next_movables)
        self.action_hist.append(action)
        self.path.append(self.agent)

        # ---------------- 3) 事件奖励：首次打通路径 ----------------
        after_reach = self._reachable_with_movables(self.agent, self.goal, self.movable)
        if (not before_reach) and after_reach:
            reward += CONNECT_BONUS

        # ---------------- 4) 势能 shaping（向目标逼近） ----------------
        phi_sp = -octile_distance(self.agent, self.goal)  # φ(s')
        reward += ALPHA_SHAPING * (GAMMA_SHAPING * phi_sp - phi_s)

        # ---------------- 5) 终止/截断 ----------------
        if self.agent == self.goal:
            reward += R_GOAL
            terminated = True

        if (not terminated) and (self.steps >= self.conf.max_steps):
            truncated = True
            reward += R_FAIL

        # 单步奖励下界裁剪（防抖）
        reward = max(reward, REWARD_MIN)

        obs = self._build_obs()
        info = {"action_mask": self._action_mask()}
        return obs, reward, terminated, truncated, info

# ------------------------------
# 简单策略：贪心朝向目标, act as baseline
# ------------------------------
def greedy_policy(env: PushPullGridEnv, info) -> int:
    ax, ay = env.agent
    gx, gy = env.goal
    # 以目标距离启发排序
    dirs = sorted(MOVES_8, key=lambda d: octile_distance((ax + d[0], ay + d[1]), (gx, gy)))
    for d in dirs:
        idx = MOVES_8.index(d)
        if info["action_mask"][idx] == 1:
            return idx
    return int(np.random.choice(8))


def test_astar_interactive(num=10):
    success_count = 0
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=0, max_steps=200))
    for i in range(num):
        obs, info = env.reset()
        if i < 5:
            env.render_path(save_path=f"astar_interactive_initial_{i+1}.png")  # 显示初始网格状态
        path, movable_ast_frame, cost = env.astar_interactive(return_movables=False)
        if i < 5:
            env.render_path(path=path, movable=movable_ast_frame, save_path=f"astar_interactive_test_path_{i+1}.png")
            
        if len(path) > 0:
            success_count += 1 if (path[-1] == env.goal) else 0
        else:
            print(f"Episode {i+1}: No path found.")
            env.render_path(path=[], movable=None, save_path=f"astar_interactive_no_path_{i+1}.png")
    print(f"Success rate: {success_count / num:.2%}")

def test_greedy_policy(num=10):
    success_count = 0
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=0, max_steps=200))
    for i in range(num):
        obs, info = env.reset()
        if i < 5:
            env.render_path(save_path=f"greedy_initial_{i+1}.png")  # 显示初始网格状态
        done = False
        while not done:
            action = greedy_policy(env, info)
            obs, r, term, trunc, info = env.step(action)
            if term or trunc:
                done = True
        if i < 5:
            env.render_path(save_path=f"greedy_test_path_{i+1}.png")
        if env.agent == env.goal:
            success_count += 1
    print(f"Success rate: {success_count / num:.2%}")

def test_astar(num: int=10):
    success_count = 0
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=0, max_steps=200))
    for i in range(num):
        obs, info = env.reset()
        if i < 5:
            env.render_path(save_path=f"astar_initial_{i+1}.png")  # 显示初始网格状态
        path, cost = env.astar()
        if i < 5:
            env.render_path(path=path, save_path=f"astar_test_path_{i+1}.png")
        if len(path) > 0:
            success_count += 1 if (path[-1] == env.goal) else 0
            env.render_path(save_path=f"astar_initial_{i+1}.png")
            env.render_path(path=path, save_path=f"astar_success_{i+1}.png")

        # else:
        #     print(f"Episode {i+1}: No path found.")
        #     env.render_path(path=[], movable=None, save_path=f"eva_no_path_{i+1}.png")
    print(f"Success rate: {success_count / num:.2%}")


if __name__ == "__main__":
    num = 100
    # test_astar(num=num)
    # test_astar_interactive(num=num) # success rate: 100.00%
    # test_greedy_policy(num=num) # success rate: 23.00%

    env = PushPullGridEnv(GridConf(H=10, W=10, seed=0, max_steps=200))
    for i in range(1):
        path, mov, cost = env.astar_interactive(return_movables=False)
        env.render_path(path=path, movable=mov, add_legend=True)

