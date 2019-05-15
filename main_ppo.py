import numpy as np
from collections import deque

import torch

import gym
from optimizers.ppo_optimizer import PPO
from policies.policy import Policy
from utils.storage_ppo import StoragePPO as Storage


n_eps = 20000
learning_rate = 3e-4
clip_param = 0.2
n_steps = 500
value_loss_coef = 0.5
entropy_coef = 0.01
alpha = 0.99
max_grad_norm = 0.5
discount = 0.99        #gamma
gae_coef = 0.95
ppo_epoch = 10
mini_batch_size = 256

seed = 42

env_name = 'CartPole-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    env = gym.make(env_name)
    env.seed(seed)

    print('New model')
    policy = Policy('actor_critic', env.observation_space.shape[0], env.action_space.n)
    policy.to(device)

    optimizer = PPO(policy, clip_param, ppo_epoch, mini_batch_size,
                value_loss_coef, entropy_coef, learning_rate,
                max_grad_norm)

    episode_rewards = deque(maxlen=50)

    for eps in range(0, n_eps + 1):
        state = env.reset()
        storage = Storage(device=device)

        policy.eval()

        episode_rewards.append(test_env(policy, gym.make(env_name)))
        if eps % 5 == 0:
            print('Avg reward', np.mean(episode_rewards))

        for step in range(n_steps):

            state = torch.FloatTensor(state).to(device)

            with torch.no_grad():
                value, action, log_prob = policy.act(state)

            next_state, reward, done, _ = env.step(action.item())

            storage.push(state, action, log_prob, value, reward, done)

            state = next_state

            if done:
                state = env.reset()

        next_state = torch.FloatTensor(next_state).to(device)
        with torch.no_grad():
            next_value = policy.get_value(next_state).detach()

        storage.compute(next_value)

        policy.train()

        value_loss, action_loss, dist_entropy = optimizer.update(storage)

        with open('metrics.csv', 'a') as metrics:
            metrics.write('{},{},{}\n'.format(value_loss, action_loss, dist_entropy))


def test_env(policy, env, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).to(device)
        value, action, log_prob = policy.act(state)
        next_state, reward, done, _ = env.step(action.item())
        state = next_state
        if vis: env.render()
        total_reward += reward
        if done: break
    return total_reward


if __name__ == '__main__':
    main()
