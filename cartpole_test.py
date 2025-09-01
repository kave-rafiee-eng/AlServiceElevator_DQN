import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    if done or truncated:
        break

env.close()