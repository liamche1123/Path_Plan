from environment import ContinuousPathPlanningEnv
from dqn_agent import DQNAgent
import numpy as np

env = ContinuousPathPlanningEnv()
agent = DQNAgent(state_size=3, action_bound=np.pi / 6)

episodes = 500
best_reward = -float("inf")
best_path = None
best_episode = None

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    for step in range(100):
        # action: angle change
        action = agent.act(state)  #  action [-π/6, π/6] angle change

        # action, new state
        next_state, reward, done, _ = env.step([action])  # 需要封装成列表

        # store experience
        agent.remember(state, action, reward, next_state, done)

        # update state
        state = next_state
        total_reward += reward

        if done:
            break

    # train
    agent.replay()

    # best path
    if total_reward > best_reward:
        best_reward = total_reward
        best_path = env.path.copy()
        best_episode = episode

    print(f"Episode {episode + 1}: Reward {total_reward:.2f}, Steps {len(env.path)}")

# save best path
env.path = best_path
env.render(best_episode)
print(f"Best path saved from Episode {best_episode} with Reward {best_reward:.2f}")
