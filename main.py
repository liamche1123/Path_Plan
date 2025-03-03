from environment import ContinuousPathPlanningEnv
from dqn_agent import DQNAgent
import numpy as np

# 初始化环境和代理
env = ContinuousPathPlanningEnv()
agent = DQNAgent(state_size=2, action_size=2)  # 动作空间改为 2（连续动作）

episodes = 500
best_reward = -float("inf")
best_path = None
best_episode = None

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    for step in range(100):  # 限制最大步数
        # 获取动作（连续动作）
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)

        # 存储经验
        agent.remember(state, action, reward, next_state, done)

        # 更新状态和总奖励
        state = next_state
        total_reward += reward

        if done:
            break

    # 训练代理
    agent.replay()

    # 记录最佳路径
    if total_reward > best_reward:
        best_reward = total_reward
        best_path = env.path
        best_episode = episode

    print(f"Episode {episode + 1}: Reward {total_reward:.2f}, Steps {len(env.path)}")

# 保存最佳路径
env.path = best_path
env.render(best_episode)
print(f"Best path saved from Episode {best_episode} with Reward {best_reward}")