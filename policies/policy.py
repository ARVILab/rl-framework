import torch.nn as nn
from .ppo.actor_critic import ActorCritic
from .dqn.dqn import DQN


class Policy(nn.Module):
    def __init__(self, policy_name, obs_shape, action_space):
        super(Policy, self).__init__()

        self.policy = None
        self.policy_name = policy_name
        if self.policy_name == 'actor_critic':
            self.policy = ActorCritic(obs_shape, action_space)
        elif self.policy_name == 'dqn':
            self.policy = DQN(obs_shape, action_space)
        else:
            raise ValueError("Name not found!")

    def forward(self, *args):
         raise NotImplementedError

    def act(self, x, epsilon=0):
        if self.policy_name == 'actor_critic':
            value, action, log_prob = self.policy(x)
            return value, action, log_prob
        elif self.policy_name == 'dqn':
            action = self.policy.act(x, epsilon)
            return action

    def eval_action(self, x, action):
        value, log_prob, entropy = self.policy.eval_action(x, action)
        return value, log_prob, entropy

    def get_value(self, x):
        if self.policy_name == 'actor_critic':
            value, _, _ = self.policy(x)
        elif self.policy_name == 'dqn':
            value = self.policy(x)  # it's q value
        return value
