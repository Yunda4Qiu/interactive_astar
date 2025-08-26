# evaluation.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import os 
import numpy as np
import torch
from ppo import PPO, set_seed, get_device
from env import PushPullGridEnv, GridConf, MOVES_8
from train import compute_prior_logits_env

# ------------------------------
# 评估工具
# ------------------------------
@torch.no_grad()
def _masked_logits(agent: PPO, obs: np.ndarray, action_mask: np.ndarray):
    s = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
    logits = agent.actor(s)
    mask = torch.as_tensor(action_mask, dtype=torch.bool, device=agent.device).unsqueeze(0)
    safe = mask.any(dim=1, keepdim=True)
    logits = torch.where(safe, logits.masked_fill(~mask, float("-inf")), logits)
    return logits

@torch.no_grad()
def select_action_eval(agent: PPO, obs: np.ndarray, action_mask: np.ndarray, greedy: bool = True) -> int:
    logits = _masked_logits(agent, obs, action_mask)

    # prior = compute_prior_logits_env(env, action_mask, beta=1.0, two_step=True) # 新
    # # stronger fusion at eval
    # logits = _masked_logits(agent, obs, action_mask) + 0.8 * torch.as_tensor(prior, device=agent.device).unsqueeze(0) # 新

    if greedy:
        if torch.isinf(logits).all():
            probs = torch.softmax(agent.actor(torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)), dim=-1)
            return int(torch.argmax(probs, dim=-1).item())
        return int(torch.argmax(logits, dim=-1).item())
    else:
        dist = torch.distributions.Categorical(logits=logits)
        return int(dist.sample().item())
    
def evaluate_agent(env: PushPullGridEnv, agent: PPO, episodes: int = 100, greedy: bool = True,
                   save_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, float]:
    if save_dir is not None: os.makedirs(save_dir, exist_ok=True)
    success, returns, steps = 0, [], []
    for i in range(episodes):
        obs, info = env.reset()
        done, ep_ret, t = False, 0.0, 0
        if save_dir is not None and (i < 5):
            env.render_path(save_path=os.path.join(save_dir, f"ep{i:03d}_initial.png"))
        while not done:
            a = select_action_eval(agent, obs, info["action_mask"], greedy=greedy)
            next_obs, r, terminated, truncated, next_info = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += r; t += 1
            obs, info = next_obs, next_info
        success += int(env.agent == env.goal)
        returns.append(ep_ret); steps.append(t)
        if save_dir is not None and (i < 5):
            env.render_path(save_path=os.path.join(save_dir, f"ep{i:03d}_final.png"))
        if verbose and ((i+1) % max(1, episodes//10) == 0):
            print(f"[Eval] {i+1}/{episodes}  Return={ep_ret:.2f}  Steps={t}  Success={env.agent==env.goal}")
    metrics = {
        "success_rate": success / episodes,
        "avg_return": float(np.mean(returns)),
        "avg_steps": float(np.mean(steps)),
    }
    if verbose:
        print("\n===== EVAL SUMMARY =====")
        print(f"Success Rate : {metrics['success_rate']:.2%}")
        print(f"Avg Return   : {metrics['avg_return']:.3f}")
        print(f"Avg Steps    : {metrics['avg_steps']:.1f}")
        print("========================\n")
    return metrics
   
# （可选）A* 参考：检查是否存在静态+movables 下的到达路径（或画出 A* 轨迹）
def evaluate_astar_interactive_baseline(env: PushPullGridEnv, episodes: int = 20, seed_offset: int = 2000,
                            save_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, float]:
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    success = 0
    costs: List[float] = []
    for i in range(episodes):
        obs, info = env.reset()
        if save_dir is not None and (i < 5):
            env.render_path(save_path=os.path.join(save_dir, f"astar_interactive_ep{i:03d}_initial.png"))
        path, final_mov, cost = env.astar_interactive(return_movables=False)
        ok = (len(path) > 0 and path[-1] == env.goal)
        success += int(ok)
        costs.append(cost if np.isfinite(cost) else 1e9)
        if save_dir is not None and (i < 5):
            if ok:
                env.render_path(path=path, movable=final_mov, save_path=os.path.join(save_dir, f"astar_interactive_ep{i:03d}.png"))
                if i == 4:
                    env.render_path(path=path, movable=final_mov, save_path=f"astar_interactive_test_path_{i}.png", add_legend=True)
            else:
                env.render_path(path=[], movable=None, save_path=os.path.join(save_dir, f"astar_interactive_ep{i:03d}_no_path.png"))
        if verbose and ((i+1) % max(1, episodes//10) == 0):
            print(f"[A*] Episode {i+1}/{episodes}  FoundPath={ok}  Cost={cost:.2f}")
    metrics = {
        "astar_interactive_success_rate": success / episodes,
        "astar_interactive_avg_cost": float(np.mean(costs)),
        "episodes": episodes
    }
    if verbose:
        print("\n===== A* SUMMARY =====")
        print(f"A* Success Rate : {metrics['astar_interactive_success_rate']:.2%}")
        print(f"A* Avg Cost     : {metrics['astar_interactive_avg_cost']:.3f}")
        print("======================\n")
    return metrics


# （可选）A* 参考：检查是否存在静态+movables 下的到达路径（或画出 A* 轨迹）
def evaluate_astar_baseline(env: PushPullGridEnv, episodes: int = 20, seed_offset: int = 2000,
                            save_dir: Optional[str] = None, verbose: bool = True) -> Dict[str, float]:
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    success = 0
    costs: List[float] = []
    for i in range(episodes):
        obs, info = env.reset()
        if save_dir is not None and (i < 5):
            env.render_path(save_path=os.path.join(save_dir, f"astar_ep{i:03d}_initial.png"))
        path, cost = env.astar()
        ok = (len(path) > 0 and path[-1] == env.goal)
        success += int(ok)
        costs.append(cost if np.isfinite(cost) else 1e9)
        if save_dir is not None and (i < 5):
            if ok:
                env.render_path(path=path, save_path=os.path.join(save_dir, f"astar_ep{i:03d}.png"))
            else:
                env.render_path(path=[], save_path=os.path.join(save_dir, f"astar_ep{i:03d}_no_path.png"))
        if verbose and ((i+1) % max(1, episodes//10) == 0):
            print(f"[A*] Episode {i+1}/{episodes}  FoundPath={ok}  Cost={cost:.2f}")
    metrics = {
        "astar_success_rate": success / episodes,
        "astar_avg_cost": float(np.mean(costs)),
        "episodes": episodes
    }
    if verbose:
        print("\n===== A* SUMMARY =====")
        print(f"A* Success Rate : {metrics['astar_success_rate']:.2%}")
        print(f"A* Avg Cost     : {metrics['astar_avg_cost']:.3f}")
        print("======================\n")
    return metrics

def load_checkpoint(path, device="cpu"):
    """
    加载 PPO checkpoint.
    兼容 PyTorch 2.6+ 的 weights_only 限制.
    """
    try:
        # PyTorch < 2.6 或者权重文件兼容时
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch < 2.6 没有 weights_only 参数
        ckpt = torch.load(path, map_location=device)
    return ckpt

if __name__ == "__main__":
    seed = 2
    set_seed(seed)
    device = get_device()
    # Env
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=seed, max_steps=200))
    # Load (fresh agent for demo)
    ckpt_path = "./checkpoints/ppo_pushpull_conv.pt"
    ckpt = load_checkpoint(ckpt_path, device=device)
    agent2 = PPO(
        in_channels=ckpt["config"]["in_channels"],
        H=ckpt["config"]["H"], W=ckpt["config"]["W"],
        action_dim=ckpt["config"]["action_dim"],
        device=device
    )
    agent2.actor.load_state_dict(ckpt["actor"])
    agent2.critic.load_state_dict(ckpt["critic"])
    agent2.actor.eval(); agent2.critic.eval()
    print("Loaded checkpoint for evaluation.")

    print("=== A* evaluation ===")
    astar_metrics = evaluate_astar_baseline(env, episodes=100, save_dir='results/astar')

    # Evaluate
    set_seed(seed)
    device = get_device()
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=seed, max_steps=200))
    print("=== Greedy evaluation ===")
    metrics_g = evaluate_agent(env, agent2, episodes=100, greedy=True,  save_dir='results/greedy', verbose=True)

    set_seed(seed)
    device = get_device()
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=seed, max_steps=200))
    print("=== Stochastic evaluation ===")
    metrics_s = evaluate_agent(env, agent2, episodes=100, greedy=False, save_dir='results/stochastic', verbose=True)

    set_seed(seed)
    device = get_device()
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=seed, max_steps=200))
    print("=== A* interactive evaluation ===")
    astar_interactive_metrics = evaluate_astar_interactive_baseline(env, episodes=100, save_dir='results/astar_interactive')

    # Print
    print("=== RESULTS ===")
    print("A*:", astar_metrics)
    print("Greedy:", metrics_g)
    print("Stochastic:", metrics_s)
    print("A* Interactive:", astar_interactive_metrics)

'''
=== RESULTS ===
A*: {'astar_success_rate': 0.04,                        'astar_avg_cost': 960000000.4687005, 'episodes': 100}
Greedy: {'success_rate': 0.35,                          'avg_return': -102.01088882285258, 'avg_steps': 133.5}
Stochastic: {'success_rate': 0.78,                      'avg_return': 15.99112632607555, 'avg_steps': 62.83}
A* Interactive: {'astar_interactive_success_rate': 1.0, 'astar_interactive_avg_cost': 13.333458864299487, 'episodes': 100}
'''