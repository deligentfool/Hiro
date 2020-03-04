import torch
import torch.nn as nn
import torch.nn.functional as F
from net import td3, value_net, policy_net
from replay_buffer import replay_buffer
import numpy as np
import random
import gym


class hiro(object):
    def __init__(self, observation_dim, goal_dim, action_dim, batch_size, learning_rate, gamma, capacity, rho, update_iter, policy_delay, epsilon_init, decay, epsilon_min, max_g, min_g, max_a, min_a, noisy_range, c, log):
        self.observation_dim = observation_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.policy_delay = policy_delay
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.max_g = max_g
        self.min_g = min_g
        self.max_a = max_a
        self.min_a = min_a
        self.noisy_range = noisy_range
        self.c = c
        self.log = log

        self.high_policy = td3(
            observation_dim=self.observation_dim,
            g_a_dim=self.goal_dim,
            batch_size=self.batch_size,
            level='high',
            learning_rate=1e-3,
            gamma=self.gamma,
            capacity=self.capacity,
            rho=self.rho,
            update_iter=self.update_iter,
            policy_delay=self.policy_delay,
            epsilon_init=self.epsilon_init,
            decay=self.decay,
            epsilon_min=self.epsilon_min,
            max_a=self.max_g,
            min_a=self.min_g,
            noisy_range=self.noisy_range,
            log=self.log
        )

        self.low_policy = td3(
            observation_dim=self.observation_dim+self.goal_dim,
            g_a_dim=self.goal_dim+self.action_dim,
            batch_size=self.batch_size,
            level='low',
            learning_rate=1e-3,
            gamma=self.gamma,
            capacity=self.capacity,
            rho=self.rho,
            update_iter=self.update_iter,
            policy_delay=self.policy_delay,
            epsilon_init=self.epsilon_init,
            decay=self.decay,
            epsilon_min=self.epsilon_min,
            max_a=self.max_a,
            min_a=self.min_a,
            noisy_range=self.noisy_range,
            log=self.log
        )

    def change_goal(self, observation):
        return self.high_policy.policy_net.act(torch.FloatTensor(np.expand_dims(observation, 0)))

    def get_goal(self, observation, next_observation, goal):
        return np.array(observation) + np.array(goal) - np.array(next_observation)

    def get_action(self, observation, goal):
        return self.low_policy.policy_net.act(torch.FloatTensor(np.expand_dims(observation + goal, 0)))

    def get_intrinsic_reward(self, observation, next_observation, goal):
        delta = np.array(observation) + np.array(goal) - np.array(next_observation)
        return - np.linalg.norm(delta, 2)

    def store_high_traj(self, observation, goal, reward, next_observation, done, state_list, action_list):
        self.high_policy.buffer.store(observation, goal, reward, next_observation, done, state_list, action_list)

    def store_low_traj(self, observation, goal, action, reward, next_observation, next_goal, done):
        self.low_policy.buffer.store(observation, goal, action, reward, next_observation, next_goal, done)

    def re_label(self):
        batch = random.sample(self.high_policy.buffer.memory, self.batch_size)
        for i in range(len(batch)):
            s_t = batch[i][0]
            s_t_c = batch[i][3]
            g = batch[i][1]
            delta = s_t_c - s_t
            candidates = [delta + np.random.randn(* delta.shape) for _ in range(8)]
            candidates.append(g)
            candidates.append(delta)
            probs = []
            for g_hat in candidates:
                prob = 0
                for j in range(self.c):
                    s = batch[i][-2][j]
                    a = batch[i][-1][j]
                    s_ = batch[i][-2][j + 1]
                    prob += - np.sum((a - self.low_policy.policy_net.act(torch.FloatTensor(s + g_hat))) ** 2)
                    g_hat = self.get_goal(s, s_, g_hat)
                probs.append(prob)
            index = np.argmax(probs)
            g = candidates[index]
        return batch

    def train_high_policy(self):
        batch = self.re_label()
        self.high_policy.train(batch)

    def train_low_policy(self):
        batch = random.sample(self.low_policy.buffer.memory, self.batch_size)
        self.low_policy.train(batch)


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    episode = 10000
    weight_reward = None
    c = 10

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    model = hiro(
        observation_dim=observation_dim,
        goal_dim=observation_dim,
        action_dim=action_dim,
        batch_size=64,
        learning_rate=1e-3,
        gamma=0.99,
        capacity=10000,
        rho=0.995,
        update_iter=10,
        policy_delay=2,
        epsilon_init=1.,
        decay=10000,
        epsilon_min=0.01,
        max_g=[1, 1, 8],
        min_g=[-1, -1, -8],
        max_a=2.,
        min_a=-2.,
        noisy_range=0.5,
        c=c,
        log=False
    )

    for i in range(episode):
        obs = env.reset()
        total_reward = 0
        c_reward = 0
        step = 0
        action_list = []
        state_list = []
        while True:
            if step % c == 0:
                goal = model.change_goal(obs)
                model.high_policy.count += 1
                high_obs = obs
                high_goal = goal
                if step != 0:
                    model.store_high_traj(high_obs, high_goal, c_reward, next_obs, done, state_list, action_list)
                c_reward = 0
                action_list = []
                state_list = []

            action = model.get_action(obs, goal)
            model.low_policy.count += 1
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            c_reward += reward
            intrinsic_reward = model.get_intrinsic_reward(obs, next_obs, goal)
            next_goal = model.get_goal(obs, next_obs, goal)
            model.store_low_traj(obs, goal, action, intrinsic_reward, next_obs, next_goal, done)
            action_list.append(action)
            state_list.append(obs)
            obs = next_obs

            if done:
                if not weight_reward:
                    weight_reward = total_reward
                else:
                    weight_reward = 0.99 * weight_reward + 0.01 * total_reward

                model.train_high_policy()
                model.train_low_policy()
                print('episode: {}  reward: {}  weight_reward: {:.2f}'.format(i+1, total_reward, weight_reward))