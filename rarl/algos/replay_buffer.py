import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, max_size):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.max_size, obs_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, obs, action, next_obs, reward, done):
        self.obs[self.ptr] = obs
        self.action[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return self.obs[idx], self.action[idx], self.next_obs[idx], self.reward[idx], self.not_done[idx]
