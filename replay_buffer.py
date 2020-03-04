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
            s, g, r, s_, done, state_list, action_list = traj
            s = np.expand_dims(s, 0)
            s_ = np.expand_dims(s_, 0)
            self.memory.append([s, g, r, s_, done, state_list, action_list])
        else:
            s, g, a, r, s_, g_, done = traj
            s = np.expand_dims(s, 0)
            s_ = np.expand_dims(s_, 0)
            self.memory.append([s, g, a, r, s_, g_, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        if self.level == 'high':
            s, g, r, s_, done, action_list = zip(* batch)
            return np.concatenate(s, 0), g, r, np.concatenate(s_, 0), done, state_list, action_list
        else:
            s, g, a, r, s_, g_, done = zip(* batch)
            return np.concatenate(s, 0), g, a, r, np.concatenate(s_, 0), g_, done

    def __len__(self):
        return len(self.memory)