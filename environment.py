import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces

class ContinuousPathPlanningEnv(gym.Env):
    def __init__(self):
        super(ContinuousPathPlanningEnv, self).__init__()

        # Map parameters
        self.map_size = 5.0
        self.start = np.array([1.0, 0.0], dtype=np.float32)
        self.goal = np.array([4.0, 4.0], dtype=np.float32)
        self.goal_radius = 0.2
        self.obstacles = [np.array([2.0, 2.0], dtype=np.float32)]

        # Agent parameters
        self.agent_position = self.start.copy()
        self.agent_heading = 0.0  # Initial heading angle (in radians)
        self.velocity = 0.1  # Fixed speed of movement
        self.path = [self.agent_position.copy()]

        # Action space: Heading angle change in radians
        self.action_space = spaces.Box(low=-np.pi/6, high=np.pi/6, shape=(1,), dtype=np.float32)

        # Observation space: Position (x, y) and heading angle
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, -np.pi]),
                                            high=np.array([self.map_size, self.map_size, np.pi]),
                                            dtype=np.float32)

    def reset(self):
        self.agent_position = self.start.copy()
        self.agent_heading = 0.0
        self.path = [self.agent_position.copy()]
        return np.array([self.agent_position[0], self.agent_position[1], self.agent_heading], dtype=np.float32)

    def step(self, action):
        # Update heading angle
        self.agent_heading += action[0]
        self.agent_heading = np.clip(self.agent_heading, -np.pi, np.pi)

        # Compute new position based on heading and velocity
        dx = self.velocity * np.cos(self.agent_heading)
        dy = self.velocity * np.sin(self.agent_heading)
        new_position = self.agent_position + np.array([dx, dy], dtype=np.float32)

        # Boundary check
        new_position = np.clip(new_position, 0.0, self.map_size)

        # Check collision with obstacles (enlarged obstacle size)
        collision = any(np.linalg.norm(new_position - obs) < 0.8 for obs in self.obstacles)  # increased from 0.5 to 0.8

        # Compute reward
        distance_old = np.linalg.norm(self.agent_position - self.goal)
        distance_new = np.linalg.norm(new_position - self.goal)

        if distance_new < self.goal_radius:
            reward = 100.0  # Reached goal
            done = True
        elif collision:
            reward = -50.0  # Collision penalty
            done = False
        else:
            reward = (distance_old - distance_new) * 10 - 0.1  # Moving towards the goal
            done = False

        # Update agent position
        self.agent_position = new_position
        self.path.append(self.agent_position.copy())

        return np.array([self.agent_position[0], self.agent_position[1], self.agent_heading], dtype=np.float32), reward, done, {}

    def render(self, episode):
        plt.figure(figsize=(5, 5))
        # Draw obstacles (enlarged size)
        for obs in self.obstacles:
            circle = plt.Circle((obs[0], obs[1]), radius=0.8, color='black', alpha=0.5, label="Obstacle" if obs is self.obstacles[0] else "")
            plt.gca().add_patch(circle)
        # Draw start and goal
        plt.scatter(self.start[0], self.start[1], c='blue', marker='o', s=200, label="Start")
        plt.scatter(self.goal[0], self.goal[1], c='red', marker='*', s=200, label="Goal")
        # Draw path
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], c='green', linestyle='-', linewidth=2, label="Path")
        plt.scatter(path[:, 0], path[:, 1], c='green', marker='.', s=100)
        # Set plot properties
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.grid(True)
        plt.legend()
        plt.title(f"Episode {episode}")
        plt.savefig(f"path_episode_{episode}.png")
        plt.close()