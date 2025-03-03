import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class ContinuousPathPlanningEnv(gym.Env):
    def __init__(self):
        super(ContinuousPathPlanningEnv, self).__init__()

        self.grid_size = 5  # 减小地图尺寸
        self.start = (1, 0)  # 固定起点
        self.goal = (4, 4)  # 固定目标点
        self.obstacles = [(2, 2)]  # 固定障碍物

        self.agent_position = self.start  # 代理初始位置
        self.path = []  # 记录路径

        self.action_space = spaces.Discrete(4)  # 4个方向：上、下、左、右
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(2,), dtype=np.int32)

    def reset(self):
        self.agent_position = self.start
        self.path = [self.agent_position]
        return np.array(self.agent_position, dtype=np.float32)

    def step(self, action):
        x, y = self.agent_position
        if action == 0:  # 上
            y = max(0, y - 1)
        elif action == 1:  # 下
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # 左
            x = max(0, x - 1)
        elif action == 3:  # 右
            x = min(self.grid_size - 1, x + 1)

        new_position = (x, y)

        # 奖励机制
        if new_position == self.goal:
            reward = 100  # 到达目标点高奖励
            done = True
        elif new_position in self.obstacles:
            reward = -50  # 撞到障碍物大惩罚
            done = False
        else:
            distance_old = np.linalg.norm(np.array(self.agent_position) - np.array(self.goal))
            distance_new = np.linalg.norm(np.array(new_position) - np.array(self.goal))
            reward = (distance_old - distance_new) * 10 - 0.1  # 靠近目标加分
            done = False

        self.agent_position = new_position
        self.path.append(self.agent_position)

        return np.array(self.agent_position, dtype=np.float32), reward, done, {}

    def render(self, episode):
        plt.figure(figsize=(5, 5))
        for obs in self.obstacles:
            plt.scatter(*obs, c='black', marker='s', s=200)  # 障碍物
        plt.scatter(*self.start, c='blue', marker='o', s=200, label="Start")  # 起点
        plt.scatter(*self.goal, c='red', marker='*', s=200, label="Goal")  # 目标点
        for step in self.path:
            plt.scatter(*step, c='green', marker='.')  # 走过的路径
        plt.legend()
        plt.savefig(f"path_episode_{episode}.png")
        plt.close()
