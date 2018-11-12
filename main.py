import time
import os
import numpy as np
from collections import deque

import torch

import gym
from optimizers.ppo import PPO
from policies.policy import Policy
from utils.storage import Storage
from utils.envs import make_env_vec


n_eps = 401
learning_rate = 3e-5
clip_param = 0.2
n_steps = 400
value_loss_coef = 0.5
entropy_coef = 0.01
alpha = 0.99
max_grad_norm = 0.5
n_processes = 16
discount = 0.99 #gamma
gae_coef = 0.95
ppo_epoch = 10
n_mini_batch = 32
epsilon = 1e-8


seed = 42
log_dir = './log'
save_path = './saved_models'

env_names = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
env_num = 2

load = False
load_eps = 5


def main():
    torch.set_num_threads(1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # For better optimization we create a vector of environments and run them in parallel
    # creating vector of environments
    envs = make_env_vec(env_names[env_num], seed, n_processes, discount, log_dir, device)

    if load:
        policy = torch.load(os.path.join(save_path, '{}-{}.pt'.format(env_names[env_num], load_eps)))

    # init policy
    policy = Policy(envs.observation_space.shape, envs.action_space)
    policy.to(device)

    # init optimizer
    agent = PPO(policy, clip_param, ppo_epoch, n_mini_batch,
                value_loss_coef, entropy_coef, learning_rate, epsilon,
                max_grad_norm)

    # init storage to save action, obs etc. to use during optimization
    storage = Storage(n_steps, n_processes, envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    storage.add_obs(obs, step=0)
    storage.to(device)

    episode_rewards = deque(maxlen=20)

    start = time.time()
    for eps in range(n_eps):
        for step in range(n_steps):

            with torch.no_grad():
                value, action, action_log_prob = policy.act(storage.obs[step])

            obs, reward, done, infos = envs.step(action)

            # ugly way to get rewards
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # we need mask so that when one of the env is done it will not be used for optimization
            masks = torch.FloatTensor([[0.0] if d else [1.0] for d in done])

            #insert new data to storage
            storage.insert(obs, action, action_log_prob, value, reward, masks)

        # for A2C we also need next state value
        with torch.no_grad():
            next_value = policy.get_value(storage.obs[-1]).detach()

        # we need to compute returns for advantage function
        storage.compute_returns(next_value, discount, gae_coef)

        # optimization
        value_loss, action_loss, dist_entropy = agent.update(storage)

        storage.after_update()

        total_num_steps = (eps + 1) * n_processes * n_steps


        # reward outputs
        if eps % 1 == 0 and len(episode_rewards) > 1:
            end = time.time()
            print("Updates {}, timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n".
                format(eps, total_num_steps,
                       int(total_num_steps / (end - start)),
                       len(episode_rewards),
                       np.mean(episode_rewards),
                       np.median(episode_rewards),
                       np.min(episode_rewards),
                       np.max(episode_rewards)))
            # print('Value loss {}. Action_loss {}\n'.format(value_loss, action_loss))

        # saveing model
        if eps % 5 == 0 and len(episode_rewards) > 1:
            print('Saving model after {} episodes'.format(eps))
            torch.save(policy, os.path.join(save_path, '{}-{}.pt'.format(env_names[env_num], eps)))


    envs.close()


if __name__ == '__main__':
    main()
