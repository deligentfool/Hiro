import numpy as np
import random
from collections import deque


class replay_buffer(object):
    def __init__(self, capacity, level):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.level = level

    def store(self, traj):
        if self.level == 'high':
            observation, goal, reward, next_observation, done, state_list, action_list = traj
            observation = np.expand_dims(observation, 0)
            next_observation = np.expand_dims(next_observation, 0)
            self.memory.append([observation, goal, reward, next_observation, done, state_list, action_list])
        else:
            observation, goal, action, reward, next_observation, next_goal, done = traj
            observation = np.expand_dims(observation, 0)
            next_observation = np.expand_dims(next_observation, 0)
            self.memory.append([observation, goal, action, reward, next_observation, next_goal, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        if self.level == 'high':
            observations, goals, rewards, next_observations, dones, action_list = zip(* batch)
            return np.concatenate(observations, 0), goals, rewards, np.concatenate(next_observations, 0), dones, state_list, action_list
        else:
            observations, goals, actions, rewards, next_observations, next_goals, dones = zip(* batch)
            return np.concatenate(observations, 0), goals, actions, rewards, np.concatenate(next_observations, 0), next_goals, dones

    def __len__(self):
        return len(self.memory)