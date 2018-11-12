import gym
import torch
import os
from policies.policy import Policy
from policies.random import RandomAgent
from utils.envs import make_env_vec

save_path = './saved_models'
env_names = ['CartPole-v0', 'CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0', 'Pendulum-v0']
env_num = 2
eps = 180

# You will get am error ImportError: sys.meta_path is None, Python is likely shutting down
# I tried to solved it but it basically the error on OpenAI side
# It doesn't affect the environment running. It just a little annoying
# Occurs when you use rendering


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    env = make_env_vec(env_names[env_num], 42, 1, None, None, device)

    policy = torch.load(os.path.join(save_path, '{}-{}.pt'.format(env_names[env_num], eps)))
    # policy = RandomAgent(env.action_space, device)

    sum_reward = 0
    obs = env.reset()

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

    env.close()
    print('{} steps, reward {}'.format(i, sum_reward))

if __name__ == '__main__':
    main()
