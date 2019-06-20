import numpy as np
from collections import deque

import torch

import gym
from optimizers.hdqn_optimizer import HDQNOptimizer

from policies.policy import Policy
from utils.storage_dqn import StorageDQN as Storage


from env.env import MountainCarEnvInherit
from env.goal import Goal


n_eps = 20000
learning_rate = 3e-3
n_steps = 500
max_grad_norm = 0.5
discount = 0.99
mini_batch_size = 256
update_epochs = 1
e_decay = 0.02

seed = 42

env_name = 'MountainCar-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_intrinsic_reward(goal, state, decimals):

    goal_round = round(goal, ndigits=decimals)
    state_round = round(state[0].item(), ndigits=decimals)

    return 1.0 if goal_round == state_round else 0.0


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    # Define goal set
    goal_object = Goal()

    env = MountainCarEnvInherit()
    env.seed(42)

    decimals = goal_object.get_decimals()

    policy = Policy('dqn', goal_object.get_env_map_size() * 2 + 1, env.action_space.n) #Controller
    target_policy = Policy("dqn", goal_object.get_env_map_size() * 2 + 1, env.action_space.n) #Controller
    meta_policy = Policy("dqn", goal_object.get_env_map_size() + 1, 10) #MetaController
    target_meta_policy = Policy("dqn", goal_object.get_env_map_size() + 1, 10) #MetaController

    policy.to(device)
    target_policy.to(device)
    target_policy.load_state_dict(policy.state_dict())

    meta_policy.to(device)
    target_meta_policy.to(device)
    target_meta_policy.load_state_dict(meta_policy.state_dict())

    optimizer_meta = HDQNOptimizer(policy, target_policy, meta_policy, target_meta_policy,
                                   mini_batch_size, discount, learning_rate, update_epochs)

    episode_rewards = deque(maxlen=50)

    get_epsilon = lambda episode: np.exp(-episode * e_decay)

    for eps in range(0, n_eps + 1):

        print(f"Game #{eps + 1}")

        state = env.reset()
        storage = Storage(device=device)
        storage_meta = Storage(device=device)

        encoded_current_state = goal_object.one_hot_current_state(state)

        episode_rewards.append(test_env(policy, gym.make(env_name), goal_object))
        if eps % 5 == 0:
            print('Avg reward', np.mean(episode_rewards))

        for step in range(n_steps):

            goal_idx = goal_object.get_random_goal_idx()
            goal = goal_object.get_goal_by_idx(goal_idx)
            encoded_goal = goal_object.one_hot_goal(goal_idx)

            total_extrinsic_reward = 0

            # while not done and not goal_reached:
            for internal_step in range(5):

                # get state and extend it with goal
                state = torch.FloatTensor(state).to(device)
                joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)
                joint_state_goal = torch.FloatTensor(joint_state_goal).to(device)

                with torch.no_grad():
                    # action = policy.act(state, get_epsilon(eps))
                    action = policy.act(joint_state_goal, 0)
                next_state, extrinsic_reward, done, _ = env.step(action.item())

                # get next_state and hand-crafted reward, after- extend it with goal
                encoded_next_state = goal_object.one_hot_current_state(next_state)
                intrinsic_reward = get_intrinsic_reward(goal, state, decimals)
                done = round(int(next_state[0]), ndigits=decimals) == round(goal, ndigits=decimals)

                joint_next_state_goal = np.concatenate([encoded_next_state, encoded_goal], axis=1)
                storage.push(joint_state_goal, action, intrinsic_reward, joint_next_state_goal, done)

                total_extrinsic_reward += extrinsic_reward

                encoded_current_state = encoded_next_state
                state = next_state

                if done:
                    break

            encoded_current_state = torch.FloatTensor(encoded_current_state).to(device)
            goal = torch.FloatTensor([goal]).to(device)
            storage_meta.push(encoded_current_state, goal, total_extrinsic_reward, encoded_next_state, done)

            encoded_current_state = encoded_current_state.cpu().detach().numpy()

            if done:
                state = env.reset()

        storage.compute()
        storage_meta.compute()

        loss = optimizer_meta.update(storage, storage_meta)

        with open('metrics.csv', 'a') as metrics:
            metrics.write('{}\n'.format(loss))


def test_env(policy, env, goal_object, vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0

    decimals = goal_object.get_decimals()
    goal_idx = goal_object.get_random_goal_idx()
    goal = goal_object.get_goal_by_idx(goal_idx)
    encoded_goal = goal_object.one_hot_goal(goal_idx)

    while not done:

        encoded_current_state = goal_object.one_hot_current_state(state)
        state = torch.FloatTensor(state).to(device)

        joint_state_goal = np.concatenate([encoded_current_state, encoded_goal], axis=1)
        joint_state_goal = torch.FloatTensor(joint_state_goal).to(device)

        action = policy.act(joint_state_goal)
        next_state, reward, done, _ = env.step(action.item())

        intrinsic_reward = get_intrinsic_reward(goal, state, decimals)

        state = next_state
        if vis: env.render()
        total_reward += (reward + intrinsic_reward)
        if done: break
    return total_reward


if __name__ == '__main__':
    main()
