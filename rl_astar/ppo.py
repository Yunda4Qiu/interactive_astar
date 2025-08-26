from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps") # for macbook, even slower than cpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ------------------------------
# Conv nets
# ------------------------------
class ConvPolicyNet(nn.Module):
    def __init__(self, in_channels: int, H: int, W: int, action_dim: int):
        super().__init__()
        self.H, self.W = H, W
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * H * W, 256), nn.ReLU(),
            nn.Linear(256, action_dim)      # logits
        )
    def forward(self, x):  # x: (B, C, H, W)
        x = self.backbone(x)
        return self.head(x)

class ConvValueNet(nn.Module):
    def __init__(self, in_channels: int, H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * H * W, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):  # x: (B, C, H, W)
        x = self.backbone(x)
        return self.head(x)

# ------------------------------
# Utils
# ------------------------------
def mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    logits: (B, A)
    mask:   (B, A) bool
    """
    if mask is None:
        return logits
    # 若某行全 0，避免 -inf 全部
    safe = mask.any(dim=1, keepdim=True)
    filled = logits.masked_fill(~mask, float("-inf"))
    return torch.where(safe, filled, logits)

def compute_advantage(gamma: float, lmbda: float, td_delta: torch.Tensor) -> torch.Tensor:
    td = td_delta.detach().cpu().numpy().reshape(-1)
    adv = 0.0
    adv_list = []
    for delta in td[::-1]:
        adv = gamma * lmbda * adv + float(delta)
        adv_list.append(adv)
    adv_list.reverse()
    return torch.as_tensor(adv_list, dtype=torch.float32, device=td_delta.device).unsqueeze(1)

# ------------------------------
# PPO 
# ------------------------------
class PPO:
    def __init__(
        self,
        in_channels: int, H: int, W: int, action_dim: int,
        actor_lr: float = 3e-4, critic_lr: float = 1e-3,
        lmbda: float = 0.95, epochs: int = 8, eps: float = 0.2, gamma: float = 0.98,
        value_coef: float = 0.5, entropy_coef: float = 0.02, value_clip: float = 0.2,
        max_grad_norm: float = 0.5, minibatch_size: int = 256,
        kl_coef_init: float = 0.2, kl_coef_final: float = 0.03, kl_anneal_steps: int = 2000,
        device: Optional[torch.device] = None,
    ):
        self.device = device or get_device()
        self.actor = ConvPolicyNet(in_channels, H, W, action_dim).to(self.device)
        self.critic = ConvValueNet(in_channels, H, W).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # PPO
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.value_clip = value_clip
        self.max_grad_norm = max_grad_norm
        self.minibatch_size = minibatch_size

        # KL prior
        self.kl_coef_init = kl_coef_init
        self.kl_coef_final = kl_coef_final
        self.kl_anneal_steps = kl_anneal_steps
        self._updates_done = 0

    @torch.no_grad()
    def take_action(self, obs: np.ndarray,
                    action_mask: Optional[np.ndarray] = None,
                    prior_logits: Optional[np.array] = None, 
                    prior_coeff: float = 1.0, 
                    greedy: bool = False) -> int:
        # obs: (C,H,W)
        s = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1,C,H,W)
        logits = self.actor(s)  # (1,A)
        mask = None
        if prior_logits is not None: 
            pl = torch.as_tensor(prior_logits, dtype=torch.float32, device=self.device).unsqueeze(0) 
            logits = logits + prior_coeff * pl   # fuse prior 
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            logits = mask_logits(logits, mask)
        if greedy:
            if torch.isinf(logits).all():
                probs = torch.softmax(self.actor(s), dim=-1)
                return int(torch.argmax(probs, dim=-1).item())
            return int(torch.argmax(logits, dim=-1).item())
        else:
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())

    def _kl_coef(self) -> float:
        t = min(self._updates_done, self.kl_anneal_steps)
        frac = 1.0 - t / max(1, self.kl_anneal_steps)
        return float(self.kl_coef_final + (self.kl_coef_init - self.kl_coef_final) * frac)

    def update(self, traj: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        期望字段：
        states: (T,C,H,W)
        actions: (T,)
        rewards: (T,)
        next_states: (T,C,H,W)
        dones: (T,)
        masks: (T,A)  bool/int
        priors: (T,A)  float  —— 采样时算好的“先验 logits”（非法动作可设 -inf）
        """
        # to tensor
        states = torch.as_tensor(traj["states"], dtype=torch.float32, device=self.device)          # (T,C,H,W)
        next_states = torch.as_tensor(traj["next_states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(traj["actions"], dtype=torch.long, device=self.device).view(-1, 1)
        rewards = torch.as_tensor(traj["rewards"], dtype=torch.float32, device=self.device).view(-1, 1)
        dones = torch.as_tensor(traj["dones"], dtype=torch.float32, device=self.device).view(-1, 1)

        T = states.shape[0]
        masks = None
        if "masks" in traj:
            masks = torch.as_tensor(traj["masks"], dtype=torch.bool, device=self.device)  # (T,A)
        priors = None
        if "priors" in traj:
            priors = torch.as_tensor(traj["priors"], dtype=torch.float32, device=self.device)  # (T,A)

        # targets and advantages
        with torch.no_grad():
            v_next = self.critic(next_states)
            v_s = self.critic(states)
            td_target = rewards + self.gamma * v_next * (1 - dones)
            td_delta = td_target - v_s
            advantage = compute_advantage(self.gamma, self.lmbda, td_delta)

        # 标准化优势
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # old distributions
        with torch.no_grad():
            old_logits = self.actor(states)  # (T,A)
            if masks is not None:
                old_logits = mask_logits(old_logits, masks)
            old_dist = torch.distributions.Categorical(logits=old_logits)
            old_log_probs = old_dist.log_prob(actions.squeeze(1)).unsqueeze(1)
            v_old = v_s  # (T,1)

        # minibatch indices
        idx_all = torch.randperm(T, device=self.device)
        mb = self.minibatch_size if T >= self.minibatch_size else T

        # logging
        last_loss = {}
        for _ in range(self.epochs):
            for start in range(0, T, mb):
                mb_idx = idx_all[start:start+mb]

                s_b = states[mb_idx]
                ns_b = next_states[mb_idx]
                a_b = actions[mb_idx]
                adv_b = advantage[mb_idx]
                td_t_b = td_target[mb_idx]
                old_logp_b = old_log_probs[mb_idx]
                v_old_b = v_old[mb_idx]
                mask_b = masks[mb_idx] if masks is not None else None
                prior_b = priors[mb_idx] if priors is not None else None

                logits = self.actor(s_b)
                if mask_b is not None:
                    logits = mask_logits(logits, mask_b)
                dist = torch.distributions.Categorical(logits=logits)

                logp = dist.log_prob(a_b.squeeze(1)).unsqueeze(1)
                ratio = torch.exp(logp - old_logp_b)
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                # value clip
                v_pred = self.critic(s_b)
                v_clip = v_old_b + (v_pred - v_old_b).clamp(-0.2, 0.2)
                v_loss1 = (v_pred - td_t_b).pow(2)
                v_loss2 = (v_clip - td_t_b).pow(2)
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                entropy = dist.entropy().mean()

                # ---- KL prior
                kl_loss = torch.tensor(0.0, device=self.device)
                if prior_b is not None:
                    prior_logits = prior_b
                    if mask_b is not None:
                        prior_logits = mask_logits(prior_logits, mask_b)
                    log_pi = torch.log_softmax(logits, dim=-1)
                    log_q  = torch.log_softmax(prior_logits, dim=-1)
                    q = torch.softmax(prior_logits, dim=-1)
                    
                    kl = (q * (log_q - log_pi)).sum(dim=1).mean()
                    kl_coef = self._kl_coef()
                    kl_loss = kl_coef * kl

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy + kl_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                # logs
                last_loss = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "kl_loss": float(kl_loss.detach().cpu().item()),
                }

        self._updates_done += 1
        return last_loss