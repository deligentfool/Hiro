import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replay_buffer import replay_buffer
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import math


class policy_net(nn.Module):
    # * deterministic actor network, output a deterministic value as the selected action
    def __init__(self, input_dim, output_dim):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, input):
        action = self.forward(input).detach().numpy()
        return action


class value_net(nn.Module):
    def __init__(self, inputs_dim, output_dim):
        super(value_net, self).__init__()
        self.input_num = len(inputs_dim)
        self.inputs_dim = inputs_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(np.sum(self.inputs_dim), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, inputs):
        x = torch.cat(inputs, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class td3(object):
    def __init__(self,observation_dim, g_a_dim, level, batch_size, learning_rate, gamma, capacity, rho, update_iter, policy_delay, epsilon_init, decay, epsilon_min, max_a, min_a, noisy_range, log):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.level = level
        self.gamma = gamma
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.policy_delay = policy_delay
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.max_a = max_a
        self.min_a = min_a
        self.noisy_range = noisy_range
        self.log = log
        self.observation_dim = observation_dim
        self.g_a_dim = g_a_dim

        self.value_net1 = value_net([self.observation_dim, self.g_a_dim], 1)
        self.value_net2 = value_net([self.observation_dim, self.g_a_dim], 1)
        self.target_value_net1 = value_net([self.observation_dim, self.g_a_dim], 1)
        self.target_value_net2 = value_net([self.observation_dim, self.g_a_dim], 1)
        self.policy_net = policy_net(self.observation_dim, self.g_a_dim)
        self.target_policy_net = policy_net(self.observation_dim, self.g_a_dim)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())
        self.target_policy_net.load_state_dict(self.policy_net.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity, level=self.level)

        self.value_optimizer1 = torch.optim.Adam(self.value_net1.parameters(), lr=self.learning_rate)
        self.value_optimizer2 = torch.optim.Adam(self.value_net2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.weight_reward = None
        self.count = 0
        self.train_count = 0
        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(- x / self.decay)
        self.writer = SummaryWriter('runs/' + self.level + '_policy')

    def soft_update(self):
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.policy_net.parameters(), self.target_policy_net.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def train(self, batch):
        value1_loss_buffer = []
        value2_loss_buffer = []
        policy_loss_buffer = []
        if self.level == 'high':
            for iter in range(self.update_iter):
                observation, goal, reward, next_observation, done, _, _= zip(* batch)

                observation = torch.FloatTensor(np.vstack(observation))
                goal = torch.FloatTensor(np.vstack(goal))
                reward = torch.FloatTensor(reward).unsqueeze(1)
                next_observation = torch.FloatTensor(np.vstack(next_observation))
                done = torch.FloatTensor(done).unsqueeze(1)

                target_next_action = self.target_policy_net.forward(next_observation)
                target_next_action = target_next_action + np.clip(np.random.randn() * self.epsilon(self.count), - self.noisy_range, self.noisy_range)
                target_next_action = torch.FloatTensor(np.clip(target_next_action.detach().numpy(), self.min_a, self.max_a)).detach()

                q_min = torch.min(self.target_value_net1.forward([next_observation, target_next_action]), self.target_value_net2.forward([next_observation, target_next_action]))
                target_q = reward + (1 - done) * self.gamma * q_min.detach()
                q1 = self.value_net1.forward([observation, goal])
                q2 = self.value_net2.forward([observation, goal])
                value_loss1 = (q1 - target_q).pow(2).mean()
                value_loss2 = (q2 - target_q).pow(2).mean()
                value1_loss_buffer.append(value_loss1.detach().item())
                value2_loss_buffer.append(value_loss2.detach().item())

                self.value_optimizer1.zero_grad()
                value_loss1.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net1.parameters(), 0.5)
                self.value_optimizer1.step()

                self.value_optimizer2.zero_grad()
                value_loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net2.parameters(), 0.5)
                self.value_optimizer2.step()

                if (iter + 1) % self.policy_delay == 0:
                    current_action = self.policy_net.forward(observation)
                    policy_loss = (- self.value_net1.forward([observation, current_action])).mean()
                    policy_loss_buffer.append(policy_loss.detach().item())

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.)
                    self.policy_optimizer.step()

                self.soft_update()
        else:
            for iter in range(self.update_iter):
                observation, goal, action, reward, next_observation, next_goal, done = zip(* batch)

                observation = torch.FloatTensor(np.vstack(observation))
                goal = torch.FloatTensor(np.vstack(goal))
                action = torch.FloatTensor(action).squeeze().unsqueeze(1)
                reward = torch.FloatTensor(reward).unsqueeze(1)
                next_observation = torch.FloatTensor(np.vstack(next_observation))
                next_goal = torch.FloatTensor(np.vstack(next_goal))
                done = torch.FloatTensor(done).unsqueeze(1)

                target_next_action = self.target_policy_net.forward(torch.cat([next_observation, next_goal], 1))
                target_next_action = target_next_action + np.clip(np.random.randn() * self.epsilon(self.count), - self.noisy_range, self.noisy_range)
                target_next_action = torch.clamp(target_next_action, self.min_a, self.max_a).detach()

                q_min = torch.min(self.target_value_net1.forward([next_observation, next_goal, target_next_action]), self.target_value_net2.forward([next_observation, next_goal, target_next_action]))
                target_q = reward + (1 - done) * self.gamma * q_min.detach()
                q1 = self.value_net1.forward([observation, goal, action])
                q2 = self.value_net2.forward([observation, goal, action])
                value_loss1 = (q1 - target_q).pow(2).mean()
                value_loss2 = (q2 - target_q).pow(2).mean()
                value1_loss_buffer.append(value_loss1.detach().item())
                value2_loss_buffer.append(value_loss2.detach().item())

                self.value_optimizer1.zero_grad()
                value_loss1.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net1.parameters(), 0.5)
                self.value_optimizer1.step()

                self.value_optimizer2.zero_grad()
                value_loss2.backward()
                torch.nn.utils.clip_grad_norm_(self.value_net2.parameters(), 0.5)
                self.value_optimizer2.step()

                if (iter + 1) % self.policy_delay == 0:
                    current_action = self.policy_net.forward(torch.cat([observation, goal], 1))
                    policy_loss = (- self.value_net1.forward([observation, goal, current_action])).mean()
                    policy_loss_buffer.append(policy_loss.detach().item())

                    self.policy_optimizer.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.)
                    self.policy_optimizer.step()

                self.soft_update()

        if self.log:
            self.writer.add_scalar('value1_loss', np.mean(value1_loss_buffer), self.train_count)
            self.writer.add_scalar('value2_loss', np.mean(value2_loss_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)

