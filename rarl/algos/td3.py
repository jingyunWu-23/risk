from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rarl.models.aca_lstm_td3 import Actor, Critic


class TD3Agent:
    def __init__(self, obs_dim, action_dim, max_action, cfg, device):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        td3_cfg = cfg["td3"]
        self.max_action = max_action
        self.actor = Actor(obs_dim, action_dim, max_action, cfg).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, max_action, cfg).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(obs_dim, action_dim, cfg).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=td3_cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=td3_cfg["critic_lr"])
        self.total_it = 0

    def select_action(self, obs):
        obs = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32, device=self.device)
        return self.actor(obs).cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        self.total_it += 1
        td3_cfg = self.cfg["td3"]
        obs, action, next_obs, reward, not_done = replay_buffer.sample(td3_cfg["batch_size"])
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        reward = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        not_done = torch.as_tensor(not_done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            noise = torch.randn_like(action) * td3_cfg["policy_noise"]
            noise = noise.clamp(-td3_cfg["noise_clip"], td3_cfg["noise_clip"])
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * td3_cfg["discount"] * target_q

        current_q1, current_q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % td3_cfg["policy_freq"] == 0:
            actor_loss = -self.critic.q1_value(obs, self.actor(obs)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target, td3_cfg["tau"])
            self._soft_update(self.critic, self.critic_target, td3_cfg["tau"])

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": None if actor_loss is None else float(actor_loss.item()),
        }

    def _soft_update(self, source, target, tau):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
