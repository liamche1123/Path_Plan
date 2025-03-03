import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces


class ContinuousPathPlanningEnv(gym.Env):
    def __init__(self):
        super(ContinuousPathPlanningEnv, self).__init__()

        # 地图参数
        self.map_size = 5.0  # 地图大小为 5x5 的连续空间
        self.start = np.array([1.0, 0.0], dtype=np.float32)  # 起点
        self.goal = np.array([4.0, 4.0], dtype=np.float32)  # 目标点
        self.goal_radius = 0.2  # 到达目标点的判定半径
        self.obstacles = [np.array([2.0, 2.0], dtype=np.float32)]  # 障碍物位置

        # 代理初始位置
        self.agent_position = self.start.copy()
        self.path = [self.agent_position.copy()]  # 记录路径

        # 动作空间：连续的两个值，表示 x 和 y 方向的移动
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观察空间：代理的当前位置 (x, y)
        self.observation_space = spaces.Box(low=0.0, high=self.map_size, shape=(2,), dtype=np.float32)

    def reset(self):
        # 重置代理位置到起点
        self.agent_position = self.start.copy()
        self.path = [self.agent_position.copy()]
        return self.agent_position

    def step(self, action):
        # 解析动作
        dx, dy = action

        # 更新代理位置
        new_position = self.agent_position + np.array([dx, dy], dtype=np.float32)

        # 边界检查，确保代理不超出地图范围
        new_position = np.clip(new_position, 0.0, self.map_size)

        # 检查是否撞到障碍物
        collision = False
        for obstacle in self.obstacles:
            if np.linalg.norm(new_position - obstacle) < 0.5:  # 障碍物半径
                collision = True
                break

        # 奖励机制
        if np.linalg.norm(new_position - self.goal) < self.goal_radius:  # 到达目标点
            reward = 100.0
            done = True
        elif collision:  # 撞到障碍物
            reward = -50.0
            done = False
        else:  # 普通移动
            distance_old = np.linalg.norm(self.agent_position - self.goal)
            distance_new = np.linalg.norm(new_position - self.goal)
            reward = (distance_old - distance_new) * 10 - 0.1  # 靠近目标加分
            done = False

        # 更新代理位置
        self.agent_position = new_position
        self.path.append(self.agent_position.copy())

        return self.agent_position, reward, done, {}

    def render(self, episode):
        plt.figure(figsize=(5, 5))
        # 绘制障碍物
        for obs in self.obstacles:
            plt.scatter(obs[0], obs[1], c='black', marker='s', s=200, label="Obstacle" if obs is self.obstacles[0] else "")
        # 绘制起点
        plt.scatter(self.start[0], self.start[1], c='blue', marker='o', s=200, label="Start")
        # 绘制目标点
        plt.scatter(self.goal[0], self.goal[1], c='red', marker='*', s=200, label="Goal")
        # 绘制路径
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], c='green', linestyle='-', linewidth=2, label="Path")
        plt.scatter(path[:, 0], path[:, 1], c='green', marker='.', s=100)
        # 设置图形属性
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.grid(True)
        plt.legend()
        plt.title(f"Episode {episode}")
        plt.savefig(f"path_episode_{episode}.png")
        plt.close()