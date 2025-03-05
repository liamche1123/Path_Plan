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
        return torch.tanh(self.fc3(x))  # 输出值范围 [-1, 1]


class DQNAgent:
    def __init__(self, state_size, action_bound):
        self.state_size = state_size
        self.action_bound = action_bound  # 角度变化范围
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.lr = 0.001  # 学习率
        self.batch_size = 32  # 批量大小
        self.memory = deque(maxlen=5000)  # 经验回放缓冲区

        # 定义模型和优化器
        self.model = DQN(state_size, 1)  # 输出一个角度变化值
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.SmoothL1Loss()  # Huber 损失函数

    def remember(self, state, action, reward, next_state, done):
        """存储经验到经验池"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            # 随机探索：在 [-1, 1] 之间采样并映射到角度变化范围
            return np.random.uniform(-1, 1) * self.action_bound
        else:
            # 使用模型预测
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_value = self.model(state_tensor).item()
            return action_value * self.action_bound  # 映射到 [-π/6, π/6]

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        # 采样小批量经验
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch])).view(-1, 1)
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).view(-1, 1)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).view(-1, 1)

        # 计算 Q 目标值
        with torch.no_grad():
            next_q_values = self.model(next_states)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算当前 Q 值
        current_q_values = self.model(states)

        # 计算损失并更新模型
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 逐步降低探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
