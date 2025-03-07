import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # output range [-1, 1]


class DQNAgent:
    def __init__(self, state_size, action_bound):
        self.state_size = state_size
        self.action_bound = action_bound  # angle range
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01  # min rate
        self.epsilon_decay = 0.995  # decay rate
        self.lr = 0.001  # learning rate
        self.batch_size = 32  # batch size
        self.memory = deque(maxlen=5000)  # buffer size

        # model and target model
        self.model = DQN(state_size, 1)  # angle change
        self.target_model = DQN(state_size, 1)  # target network
        self.target_model.load_state_dict(self.model.state_dict())  # initialize target network
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss function

        # target network update frequency
        self.target_update_freq = 10
        self.update_counter = 0

    def remember(self, state, action, reward, next_state, done):
        """experience replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """action choice"""
        if np.random.rand() <= self.epsilon:
            # randomly explore：
            return np.random.uniform(-1, 1) * self.action_bound
        else:
            # predict
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_value = self.model(state_tensor).item()
            return action_value * self.action_bound  #  [-π/6, π/6]

    def replay(self):
        """replay"""
        if len(self.memory) < self.batch_size:
            return

        # batch sample
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch])).view(-1, 1)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).view(-1, 1)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).view(-1, 1)

        # target Q-Value
        with torch.no_grad():
            next_q_values = self.target_model(next_states)  # use target network
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # current Q-Value
        current_q_values = self.model(states)

        # Loss function
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decline the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
