import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveContextAttention(nn.Module):
    def __init__(self, ego_dim, sv_feature_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(ego_dim, attention_dim)
        self.key = nn.Linear(sv_feature_dim, attention_dim)
        self.value = nn.Linear(sv_feature_dim, attention_dim)

    def forward(self, ego, sv_features):
        q = self.query(ego).unsqueeze(1)
        k = self.key(sv_features)
        v = self.value(sv_features)
        scores = torch.sum(q * k, dim=-1) / (k.shape[-1] ** 0.5)
        valid_mask = torch.sum(torch.abs(sv_features), dim=-1) > 1e-6
        scores = scores.masked_fill(~valid_mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(valid_mask, weights, torch.zeros_like(weights))
        weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-6)
        context = torch.sum(weights.unsqueeze(-1) * v, dim=1)
        return context, weights


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action, cfg):
        super().__init__()
        self.max_action = max_action
        self.max_svs = cfg["observation"]["max_surrounding_vehicles"]
        self.ego_dim = 8
        self.sv_feature_dim = 5
        self.risk_dim = 4
        hidden_dim = cfg["td3"]["hidden_dim"]
        attention_dim = cfg["td3"]["attention_dim"]
        lstm_hidden_dim = cfg["td3"]["lstm_hidden_dim"]

        self.attention = AdaptiveContextAttention(self.ego_dim, self.sv_feature_dim, attention_dim)
        self.lstm = nn.LSTM(input_size=self.ego_dim + attention_dim + self.risk_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        ego, svs, risk = self._split_obs(obs)
        context, _ = self.attention(ego, svs)
        x = torch.cat([ego, context, risk], dim=-1).unsqueeze(1)
        x, _ = self.lstm(x)
        x = x[:, -1]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.out(x))

    def _split_obs(self, obs):
        ego = obs[:, : self.ego_dim]
        sv_start = self.ego_dim
        sv_end = sv_start + self.max_svs * self.sv_feature_dim
        svs = obs[:, sv_start:sv_end].reshape(obs.shape[0], self.max_svs, self.sv_feature_dim)
        risk = obs[:, sv_end : sv_end + self.risk_dim]
        return ego, svs, risk


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, cfg):
        super().__init__()
        hidden_dim = cfg["td3"]["hidden_dim"]
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs, action):
        return self.q1(torch.cat([obs, action], dim=-1))
