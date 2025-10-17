import gymnasium as gym
import numpy as np
import torch
from cma import CMAEvolutionStrategy
import matplotlib.pyplot as plt
import os

class CMAESTrainer:
    def __init__(self, env_name="CartPole-v1"):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]  # 4 for CartPole
        self.action_space = self.env.action_space.n  # 2 for CartPole
        self.num_params = self.observation_space * self.action_space
        
        # CMA-ES parameters
        self.sigma = 0.5  # Initial step size
        self.population_size = 16  # Population size
        self.max_iterations = 100  # Maximum number of iterations
        
        # Initialize CMA-ES optimizer
        self.es = CMAEvolutionStrategy(
            x0=np.zeros(self.num_params),  # Initial solution
            sigma0=self.sigma,
            inopts={'popsize': self.population_size}
        )

        self.training_history = {
            'best_fitness': [],
            'mean_fitness': [],
            'generation': []
        }

    def evaluate_policy(self, weights, num_episodes=5):
        """Evaluate a policy (weights) over several episodes."""
        total_rewards = []
        weights_matrix = weights.reshape(self.observation_space, self.action_space)
        
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Get action using the policy
                action_scores = np.dot(obs, weights_matrix)
                action = int(np.argmax(action_scores))
                
                # Take step in environment
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards)

    def train(self):
        """Train the model using CMA-ES."""
        best_fitness = float('-inf')
        best_weights = None
        iteration = 0
        
        while not self.es.stop() and iteration < self.max_iterations:
            solutions = self.es.ask()
            fitnesses = []
            
            for weights in solutions:
                reward = self.evaluate_policy(weights)  # Remove the [0] indexing
                fitnesses.append(reward)
            
            self.es.tell(solutions, [-f for f in fitnesses])  # CMA-ES minimizes
            
            generation_best = max(fitnesses)
            generation_mean = sum(fitnesses) / len(fitnesses)
            
            # Store training metrics
            self.training_history['best_fitness'].append(generation_best)
            self.training_history['mean_fitness'].append(generation_mean)
            self.training_history['generation'].append(iteration)
            
            if generation_best > best_fitness:
                best_fitness = generation_best
                best_weights = solutions[fitnesses.index(generation_best)]
                
            print(f"Generation {iteration}: Best Fitness = {generation_best:.2f}, Mean Fitness = {generation_mean:.2f}")
            iteration += 1
        
        # Create and save the convergence plot
        self.plot_training_history()
        
        # Save training history
        np.save('training_history.npy', self.training_history)
        
        # Save as numpy array instead of torch tensor
        np.save('cmaes_model.npy', {
            'weights': best_weights,
            'fitness': best_fitness,
            'shape': (self.observation_space, self.action_space)
        })
        print("Model saved to cmaes_model.npy")
        
        return best_weights, best_fitness

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_history['generation'], 
                self.training_history['best_fitness'], 
                label='Best Fitness', 
                color='blue')
        plt.plot(self.training_history['generation'], 
                self.training_history['mean_fitness'], 
                label='Mean Fitness', 
                color='orange', 
                alpha=0.7)
        
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Episode Return)')
        plt.title('CMA-ES Training Convergence on CartPole-v1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/training_convergence.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    trainer = CMAESTrainer()
    best_weights, best_fitness = trainer.train()
    
    print("\nTesting final model...")
    from test_model import CMAESAgent
    agent = CMAESAgent("CartPole-v1")
    agent.load_model("cmaes_model.npy")
    agent.test_model(num_episodes=5)