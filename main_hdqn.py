import numpy as np
from collections import deque

import torch

import gym
from optimizers.dqn_optimizer import DQNOptimizer

from policies.policy import Policy
from utils.storage_dqn import StorageDQN as Storage


from envs.env import MountainCarEnvInherit
from envs.goal import Goal


n_eps = 20000
learning_rate = 3e-3
n_steps = 500
max_grad_norm = 0.5
discount = 0.99
mini_batch_size = 256
update_epochs = 1
e_decay = 0.02
target_policy_update = 5

seed = 42

env_name = 'MountainCar-v0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

goal_object = Goal()


def get_intrinsic_reward(goal, state):

    state_round = round(state[0].item(), 1)
    return 1.0 if goal == goal_object.get_goal(state_round) else 0.0

def goal_wrapper(meta_policy, state, goal_object, epsilon):
    if np.random.rand() > epsilon:

        q_value = meta_policy.get_value(torch.FloatTensor(state).to(device))

        goal = torch.clamp(q_value.argmax().unsqueeze(0), min=-1.2, max=0.6)
        goal = goal.detach().cpu().item()
        goal = round(goal, goal_object.get_decimals())

        goal_idx = goal_object.get_idx_by_goal(goal)
    else:

        goal_idx = goal_object.get_random_goal_idx()
        goal = goal_object.get_goal_by_idx(goal_idx)
        goal = round(goal, goal_object.get_decimals())


    encoded_goal = goal_object.one_hot_goal(goal_idx)

    return goal, encoded_goal

def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)

    env = MountainCarEnvInherit()
    env.seed(42)

    meta_policy = Policy('dqn', env.observation_space.shape[0], goal_object.get_size()) 
    target_meta_policy = Policy('dqn', env.observation_space.shape[0], goal_object.get_size()) 

    policy = Policy('dqn', env.observation_space.shape[0] + 1, env.action_space.n) 
    target_policy = Policy('dqn', env.observation_space.shape[0] + 1, env.action_space.n) 

    meta_policy.to(device)
    target_meta_policy.to(device)
    target_meta_policy.load_state_dict(meta_policy.state_dict())

    policy.to(device)
    target_policy.to(device)
    target_policy.load_state_dict(policy.state_dict())

    optimizer_meta_policy = DQNOptimizer(meta_policy, target_meta_policy, mini_batch_size, discount, learning_rate, update_epochs)

    optimizer_policy = DQNOptimizer(policy, target_policy, mini_batch_size, discount, learning_rate, update_epochs)

    episode_rewards = deque(maxlen=50)

    get_epsilon = lambda episode: np.exp(-episode * e_decay)


    for eps in range(0, n_eps + 1):

        if eps % 50 == 0:
            episode_rewards.append(test_env(meta_policy, policy, MountainCarEnvInherit()))
            print('Avg reward', np.mean(episode_rewards))

        storage = Storage(device=device)
        storage_meta = Storage(device=device)
        print('Game', eps)

        state0 = env.reset()
        state = state0.copy()
        state = torch.FloatTensor(state).to(device)
        for step in range(10):

            extrinsic_reward = 0
            goal = meta_policy.act(state, get_epsilon(eps))

            # while not done:
            for internal_step in range(60):

                joint_state = torch.FloatTensor(np.concatenate([state.cpu().numpy(), [goal.item()]], axis=0)).to(device)

                with torch.no_grad():
                    action = policy.act(joint_state, get_epsilon(eps))

                next_state, reward, done, _ = env.step(action.item())

                intrinsic_reward = get_intrinsic_reward(goal, next_state)
                goal_reached = True if intrinsic_reward else False

                joint_next_state = np.concatenate([next_state, [goal.item()]], axis=0)
                storage.push(joint_state, action, intrinsic_reward, joint_next_state, done)

                extrinsic_reward += reward

                state = next_state
                state = torch.FloatTensor(state).to(device)

                if goal_reached or done:
                    break

            goal = torch.LongTensor([goal]).to(device)
            storage_meta.push(torch.FloatTensor(state0).to(device), goal, extrinsic_reward, next_state, done)

            if done:
                break

        storage.compute()
        storage_meta.compute()

        loss_meta = optimizer_meta_policy.update(storage_meta)
        loss = optimizer_policy.update(storage)

        if eps % target_policy_update:
            target_meta_policy.load_state_dict(meta_policy.state_dict())
            target_policy.load_state_dict(policy.state_dict())

        with open('metrics.csv', 'a') as metrics:
            metrics.write('{},{}\n'.format(loss_meta, loss))


def test_env(meta_policy, policy, env, vis=True):

    state0 = env.reset()
    state = state0.copy()
    state = torch.FloatTensor(state).to(device)
    for step in range(10):

        extrinsic_reward = 0
        goal = meta_policy.act(state)

        print('Step {}, Goal {}'.format(step, goal.item()))

        done = False
        total_intrinsic_reward = 0

        # while not done:
        for internal_step in range(60):

            if vis: env.render()

            joint_state = torch.FloatTensor(np.concatenate([state.cpu().numpy(), [goal.item()]], axis=0)).to(device)

            with torch.no_grad():
                action = policy.act(joint_state)

            next_state, reward, done, _ = env.step(action.item())

            intrinsic_reward = get_intrinsic_reward(goal, next_state)
            goal_reached = True if intrinsic_reward else False

            extrinsic_reward += reward
            total_intrinsic_reward += intrinsic_reward

            state = next_state
            state = torch.FloatTensor(state).to(device)

            if goal_reached or done:
                break

        if done:
            break

    print('Intrinsic reward', total_intrinsic_reward)
    env.close()

    return extrinsic_reward


if __name__ == '__main__':
    main()
