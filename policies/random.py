import torch
import numpy as np

class RandomAgent(object):

    def __init__(self, action_space, device):
        self.action_space = action_space
        self.device = device

    def act(self, obs):
        action = self.action_space.sample()
        action = torch.from_numpy(np.array([[action]]))
        action.to(self.device)
        return [], action, []
