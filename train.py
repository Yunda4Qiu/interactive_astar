from __future__ import annotations
import os
import numpy as np
import torch
from typing import Dict, Optional
import matplotlib.pyplot as plt

from env import PushPullGridEnv, GridConf, MOVES_8
from ppo import PPO, set_seed, get_device

# ------------------------------
# 计算先验 logits
# ------------------------------
def compute_prior_logits_env(env: PushPullGridEnv, action_mask: np.ndarray, beta: float = 1.0, two_step: bool = True) -> np.ndarray:
    priors = np.full((len(MOVES_8),), -np.inf, dtype=np.float32)
    movtuple0 = tuple(sorted(env.movable))
    for a, delta in enumerate(MOVES_8):
        if action_mask[a] == 0:
            continue
        pack1 = env.simulate_micro_step(env.agent, movtuple0, delta)
        if pack1 is None:
            continue
        cost = pack1[2]  # c + h
        if two_step:
            best2 = np.inf
            agent2, mov2 = pack1[0], tuple(sorted(pack1[1]))
            for d2 in MOVES_8:
                pack2 = env.simulate_micro_step(agent2, mov2, d2)
                if pack2 is None:
                    continue
                best2 = min(best2, pack2[2])
            if np.isfinite(best2):
                cost += best2
        priors[a] = -beta * float(cost)
    return priors

# ------------------------------
# 训练
# ------------------------------
def train(env: PushPullGridEnv, agent: PPO, num_episodes: int = 3000, prior_beta: float = 1.0, two_step: bool = True):
    import tqdm as tq
    returns = []
    for ep in tq.tqdm(range(num_episodes), desc="Training"):
        obs, info = env.reset()
        traj = {"states":[], "actions":[], "next_states":[], "rewards":[], "dones":[], "masks":[], "priors":[]}
        done = False
        ep_ret = 0.0
        while not done:
            prior_logits = compute_prior_logits_env(env, info["action_mask"], beta=prior_beta, two_step=two_step)
            prior_coeff  = max(0.2, 1.0 - ep / 1000.0)  # decay from 1.0 to 0.2 over first 1000 eps
            a = agent.take_action(obs, action_mask=info["action_mask"], prior_logits=prior_logits, prior_coeff=prior_coeff, greedy=False)
            next_obs, r, terminated, truncated, next_info = env.step(a)
            done = bool(terminated or truncated)

            traj["states"].append(obs)
            traj["actions"].append(a)
            traj["next_states"].append(next_obs)
            traj["rewards"].append(r)
            traj["dones"].append(float(done))
            traj["masks"].append(info["action_mask"].astype(bool))
            traj["priors"].append(prior_logits)

            obs, info = next_obs, next_info
            ep_ret += r

        for k in traj:
            traj[k] = np.array(traj[k])
        logs = agent.update(traj)
        returns.append(ep_ret)

        if (ep+1) % 50 == 0:
            print(f"[Train] Ep {ep+1}/{num_episodes}  Return={ep_ret:.2f}  {logs}")

    return returns

# ------------------------------
# Save / Load
# ------------------------------
def save_checkpoint(agent: PPO, path: str, env: PushPullGridEnv, returns=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "actor_opt": agent.actor_optimizer.state_dict(),
        "critic_opt": agent.critic_optimizer.state_dict(),
        "config": {
            "in_channels": 4,
            "H": env.conf.H, "W": env.conf.W,
            "action_dim": env.action_space.n,
        },
        "returns": returns if returns is not None else [],
    }, path)
    print(f"Saved to {path}")

def load_checkpoint(path: str, device=None) -> Dict:
    ckpt = torch.load(path, map_location=device or get_device())
    return ckpt

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    set_seed(0)
    device = get_device()

    # Env
    env = PushPullGridEnv(GridConf(H=10, W=10, seed=0, max_steps=200))
    C, H, W = 4, env.conf.H, env.conf.W
    A = env.action_space.n

    # Agent
    agent = PPO(
        in_channels=C, H=H, W=W, action_dim=A,
        actor_lr=3e-4, critic_lr=1e-3,
        lmbda=0.95, epochs=8, eps=0.2, gamma=0.98,
        value_coef=0.5, entropy_coef=0.02, value_clip=0.2,
        max_grad_norm=0.5, minibatch_size=256,
        kl_coef_init=0.2, kl_coef_final=0.03, kl_anneal_steps=2000,
        device=device,
    )

    # Train
    returns = train(env, agent, num_episodes=2000, prior_beta=1.0, two_step=True)

    # Save
    ckpt_path = "./checkpoints/ppo_pushpull_conv.pt"
    save_checkpoint(agent, ckpt_path, env, returns=returns)


