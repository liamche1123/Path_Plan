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
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.lr = 0.001  # 学习率
        self.batch_size = 32  # 批量大小
        self.memory = deque(maxlen=5000)  # 经验回放缓冲区

        # 定义模型和优化器
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()  # 损失函数

    def remember(self, state, action, reward, next_state, done):
        """存储经验到经验池"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            # 随机动作（连续空间）
            return np.random.uniform(-1, 1, size=self.action_size)
        else:
            # 使用模型预测动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_values = self.model(state_tensor)
            return action_values.squeeze(0).numpy()

    def replay(self):
        """经验回放训练"""
        if len(self.memory) < self.batch_size:
            return

        # 从经验池中随机采样
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch]))
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch]))
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch]))

        # 计算目标值
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 计算当前 Q 值
        current_q_values = self.model(states)
        predicted_q_values = torch.sum(current_q_values * actions, dim=1)

        # 计算损失并更新模型
        loss = self.criterion(predicted_q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay