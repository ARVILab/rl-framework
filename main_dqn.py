import numpy as np
from collections import deque

import torch

import gym
from optimizers.dqn_optimizer import DQNOptimizer
from policies.policy import Policy
from utils.storage_dqn import StorageDQN as Storage


n_eps = 20000
learning_rate = 3e-3
n_steps = 500
max_grad_norm = 0.5
discount = 0.99
mini_batch_size = 256
update_epochs = 1
target_policy_update = 5
e_decay = 0.02

seed = 42

env_name = 'CartPole-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    env = gym.make(env_name)
    env.seed(42)

    print('New model')
    policy = Policy('dqn', env.observation_space.shape[0], env.action_space.n)
    target_policy = Policy('dqn', env.observation_space.shape[0], env.action_space.n)
    policy.to(device)
    target_policy.to(device)
    target_policy.load_state_dict(policy.state_dict())
    optimizer = DQNOptimizer(policy, target_policy, mini_batch_size, discount, learning_rate, update_epochs)

    episode_rewards = deque(maxlen=50)

    get_epsilon = lambda episode: np.exp(-episode * e_decay)

    for eps in range(0, n_eps + 1):
        state = env.reset()
        storage = Storage(device=device)

        episode_rewards.append(test_env(target_policy, gym.make(env_name)))
        if eps % 5 == 0:
            print('Avg reward', np.mean(episode_rewards))

        for step in range(n_steps):

            state = torch.FloatTensor(state).to(device)

            with torch.no_grad():
                action = policy.act(state, get_epsilon(eps))

            next_state, reward, done, _ = env.step(action.item())

            storage.push(state, action, reward, next_state, done)

            state = next_state

            if done:
                state = env.reset()

        storage.compute()

        loss = optimizer.update(storage)

        if eps % target_policy_update:
            target_policy.load_state_dict(policy.state_dict())

        with open('metrics.csv', 'a') as metrics:
            metrics.write('{}\n'.format(loss))


def test_env(policy, env, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).to(device)
        action = policy.act(state)
        next_state, reward, done, _ = env.step(action.item())
        state = next_state
        if vis: env.render()
        total_reward += reward
        if done: break
    env.close()
    return total_reward


if __name__ == '__main__':
    main()
