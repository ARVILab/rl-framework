import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class StoragePPO(object):
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.masks = []
        self.returns = []
        self.advantages = []

        self.device = device

        self.gamma = 0.99     # discount
        self.tau = 0.95       # gae_coef

    def push(self, state, action, log_prob, value, reward, done):
        mask = torch.FloatTensor([[0.0] if done else [1.0]]).to(self.device)
        action = action.unsqueeze(0)
        value = value.unsqueeze(0).to(self.device)
        log_prob = log_prob.unsqueeze(0).to(self.device)
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(1).to(self.device)
        state = state.unsqueeze(0)

        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.masks.append(mask)

    def compute(self, next_value):
        self.values.append(next_value.unsqueeze(0))
        self.masks.append(self.masks[-1].clone())

        gae = 0
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[step]
            gae = delta + self.gamma * self.tau * self.masks[step + 1] * gae
            self.returns.insert(0, gae + self.values[step])

        del self.values[-1]
        del self.masks[-1]
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.log_probs = torch.cat(self.log_probs)
        self.values = torch.cat(self.values)
        self.returns = torch.cat(self.returns)
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

    def sample(self, mini_batch_size):
        batch_size = self.states.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield self.states[indices, :], self.actions[indices, :], self.log_probs[indices, :], \
                  self.returns[indices, :], self.advantages[indices, :]
