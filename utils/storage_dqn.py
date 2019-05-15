import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np


class StorageDQN(object):
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.next_states = []
        self.device = device

    def push(self, state, action, reward, next_state, done):
        mask = torch.FloatTensor([[0.0] if done else [1.0]]).to(self.device)
        action = action.unsqueeze(0)
        reward = torch.FloatTensor(np.array([reward])).unsqueeze(1).to(self.device)
        state = state.unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.masks.append(mask)
        self.next_states.append(next_state)

    def compute(self, ):
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        self.rewards = torch.cat(self.rewards)
        self.masks = torch.cat(self.masks)
        self.next_states = torch.cat(self.next_states)

    def sample(self, mini_batch_size):
        batch_size = self.states.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False)
        for indices in sampler:
            yield self.states[indices, :], self.actions[indices, :], self.rewards[indices, :], \
                  self.next_states[indices, :], self.masks[indices, :]
