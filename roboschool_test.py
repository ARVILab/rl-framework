import gym
import torch
import os
from policies.policy import Policy
from policies.random import RandomAgent
from utils.envs import make_env_vec, get_vec_normalize

import numpy as np

save_path = './saved_models'
env_names = ['RoboschoolAnt-v1', 'RoboschoolHumanoidFlagrun-v1', 'RoboschoolWalker2d-v1',
             'RoboschoolHalfCheetah-v1', 'RoboschoolHopper-v1', 'RoboschoolReacher-v1', 'RoboschoolHumanoid-v1',
             'RoboschoolHumanoid-v1']
env_num = -1
eps = 385

# You will get an error ImportError: sys.meta_path is None, Python is likely shutting down
# I tried to solved it but it basically the error on OpenAI side
# It doesn't affect the environment running. It just a little annoying
# Occurs when you use rendering


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = make_env_vec(env_names[env_num], 42, 1, None, None, device)

    policy, ob_rms = torch.load(os.path.join(save_path, '{}-{}.pt'.format(env_names[env_num], eps)),
                                       map_location=lambda storage, loc: storage)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    num_runs = 100
    rewards = []
    for _ in range(num_runs):
        obs = env.reset()
        sum_reward = 0
        i = 0
        while True:

            with torch.no_grad():
                value, action, action_log_prob = policy.act(obs)

            env.render()
            obs, reward, done, info = env.step(action)
            sum_reward += reward
            if done[0]: #because env wrapper returns array
                break
            i += 1

        print('{} steps, reward {}'.format(i, sum_reward.item()))
        rewards.append(sum_reward)

    print('Mean reward {}'.format(np.mean(rewards)))
    env.close()

if __name__ == '__main__':
    main()
