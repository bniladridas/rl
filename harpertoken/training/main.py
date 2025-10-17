import gymnasium as gym
import numpy as np
import torch
from cma import CMAEvolutionStrategy

class CMAESAgent:
    def __init__(self, env_name):
        self.env = gym.make(env_name)
        self.num_params = self.env.action_space.n
        self.es = CMAEvolutionStrategy(self.num_params * [0], 0.5)

    def train(self, num_iterations=50):
        print("Starting training...")
        for iteration in range(num_iterations):
            params = self.es.ask()
            rewards = []
            for param in params:
                total_reward = self.run_episode(param)
                rewards.append(total_reward)
            self.es.tell(params, rewards)
            
            mean_reward = np.mean(rewards)
            print(f"Iteration {iteration + 1}/{num_iterations}, Mean Reward: {mean_reward:.2f}")

    def run_episode(self, param):
        obs, _ = self.env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = np.argmax(param)
            obs, reward, terminated, truncated, _ = self.env.step(action)
            total_reward += reward
            
        return total_reward

    def save_model(self, path):
        try:
            torch.save(self.es, path, pickle_protocol=4)
            print(f"Model saved to {path}")
        except Exception as e:
            print(f"Error saving model: {e}")

if __name__ == "__main__":
    # Train the model
    print("Initializing agent...")
    agent = CMAESAgent("CartPole-v1")
    agent.train(num_iterations=50)
    agent.save_model("cmaes_model.pth")
