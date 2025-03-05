from environment import ContinuousPathPlanningEnv
from dqn_agent import DQNAgent
import numpy as np

# 初始化环境
env = ContinuousPathPlanningEnv()
agent = DQNAgent(state_size=3, action_bound=np.pi / 6)  # 适应角度变化

episodes = 500
best_reward = -float("inf")
best_path = None
best_episode = None

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):  # 限制最大步数，防止无限循环
        # 获取动作（角度变化）
        action = agent.act(state)  # 这里 action 是 [-π/6, π/6] 之间的角度变化

        # 采取动作，获取新的状态
        next_state, reward, done, _ = env.step([action])  # 需要封装成列表

        # 存储经验
        agent.remember(state, action, reward, next_state, done)

        # 更新状态
        state = next_state
        total_reward += reward

        if done:
            break

    # 训练代理
    agent.replay()

    # 记录最佳路径
    if total_reward > best_reward:
        best_reward = total_reward
        best_path = env.path.copy()
        best_episode = episode

    print(f"Episode {episode + 1}: Reward {total_reward:.2f}, Steps {len(env.path)}")

# 保存最佳路径
env.path = best_path
env.render(best_episode)
print(f"Best path saved from Episode {best_episode} with Reward {best_reward:.2f}")
