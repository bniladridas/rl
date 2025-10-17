import gymnasium as gym
import pandas as pd


def generate_cartpole_data(num_episodes=100):
    env = gym.make("CartPole-v1")
    data = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, info = env.step(action)
            data.append((obs, action, reward, next_obs))
            obs = next_obs

    env.close()
    return data


if __name__ == "__main__":
    data = generate_cartpole_data(num_episodes=100)
    columns = ["observation", "action", "reward", "next_observation"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("cartpole_data.csv", index=False)
    print("Data generated and saved to cartpole_data.csv")
