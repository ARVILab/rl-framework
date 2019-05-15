import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=1, keepdim=True)


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
              nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0),
              gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        action_probs = F.softmax(x, dim=-1)
        return FixedCategorical(probs=action_probs)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size=16, nn='resnet'):
        super(ActorCritic, self).__init__()

        self.dist_layer = Categorical(hidden_size, num_outputs)

        if nn == 'resnet':
            self.base = ResNet(num_inputs, hidden_size=hidden_size)
        elif nn == 'mlp':
            self.base = MLP(num_inputs, hidden_size=hidden_size)

    def forward(self, x):
        value, actor_features = self.base(x)

        dist = self.dist_layer(actor_features)

        action = dist.sample()
        log_prob = dist.log_probs(action)
        return value, action, log_prob

    def eval_action(self, x, action):
        value, actor_features = self.base(x)

        dist = self.dist_layer(actor_features)

        log_prob = dist.log_probs(action)
        entropy = dist.entropy().mean()
        return value, log_prob, entropy


class MLP(nn.Module):
    def __init__(self, num_inputs, hidden_size=64):
        super(MLP, self).__init__()


        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            init_(nn.Linear(hidden_size, 1))
        )

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        value = self.critic(x)
        action_features = self.actor(x)
        return value, action_features

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, downsample=None):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.fc(out)
        out = F.relu(out)
        out = self.bn(out)

        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, num_inputs, hidden_size=64, layers=[2, 2, 2]):
        super(ResNet, self).__init__()

        block = ResidualBlock

        self.fc = nn.Linear(num_inputs, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.layer1 = self.make_layer(block, hidden_size, layers[0])
        self.layer2 = self.make_layer(block, hidden_size, layers[1], True)
        self.layer3 = self.make_layer(block, hidden_size, layers[2], True)

        self.pi = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def make_layer(self, block, hidden_size, blocks, use_downsample=False):
        downsample = None
        if use_downsample:
            downsample = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size)
            )
        layers = []
        layers.append(block(hidden_size, downsample))
        for i in range(1, blocks):
            layers.append(block(hidden_size))

        return nn.Sequential(*layers)

    @property
    def output_size(self):
        return self._hidden_size

    def forward(self, x):
        if self.training:
            out = self.fc(x)
        else:
            out = self.fc(x.unsqueeze(0))
        out = self.bn(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        actor_features = self.pi(out)
        value = self.v(out)

        return value, actor_features
