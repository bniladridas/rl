from huggingface_hub import ModelHubMixin
import numpy as np
import gymnasium as gym
import atexit

class CMAESAgent(ModelHubMixin):
    """CartPole agent trained using CMA-ES optimization"""
    
    def __init__(self, env_name):
        super().__init__()
        try:
            self.env = gym.make(env_name, render_mode="human")
        except Exception as e:
            print(f"Warning: Could not create environment with human rendering: {e}")
            self.env = gym.make(env_name)
            
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.num_params = self.observation_space * self.action_space
        atexit.register(self.cleanup)

    def _save_pretrained(self, save_directory):
        """Save model weights"""
        save_path = f"{save_directory}/model.npy"
        np.save(save_path, {
            'weights': self.weights,
            'observation_space': self.observation_space,
            'action_space': self.action_space
        })

    @classmethod
    def _from_pretrained(cls, model_id, **kwargs):
        """Load model weights"""
        model = cls("CartPole-v1")
        weights_path = f"{model_id}/model.npy"
        model.load_model(weights_path)
        return model

    def cleanup(self):
        """
        Cleanup method to ensure environment is closed properly
        """
        try:
            if hasattr(self, 'env'):
                self.env.close()
        except Exception:
            pass

    def load_model(self, path):
        """
        Load the model parameters from the specified path.
        """
        print(f"Loading model from {path}")
        try:
            loaded_data = np.load(path, allow_pickle=True).item()
            self.weights = loaded_data['weights']
            
            if self.weights.ndim == 1:
                self.weights = self.weights.reshape(self.observation_space, self.action_space)
            
            print(f"Model parameters shape: {self.weights.shape}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def get_action(self, observation):
        """
        Get action from the current observation using the loaded model.
        """
        observation = np.array(observation, dtype=np.float32)
        
        # Compute action scores using the weight matrix
        action_scores = np.dot(observation, self.weights)
        
        # Add small random noise to break ties
        action_scores += np.random.randn(*action_scores.shape) * 1e-5
        
        # Return the action with highest score
        return int(np.argmax(action_scores))

    def test_model(self, num_episodes=5, render=True):
        """
        Test the model over a specified number of episodes.
        """
        if not hasattr(self, 'weights'):
            print("No model loaded. Please load a model first.")
            return

        print(f"\nTesting model for {num_episodes} episodes...")
        total_rewards = []
        
        try:
            for episode in range(num_episodes):
                obs, _ = self.env.reset()
                episode_reward = 0
                done = False
                step = 0
                
                while not done:
                    action = self.get_action(obs)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    episode_reward += reward
                    step += 1
                    done = terminated or truncated
                    
                    if render:
                        self.env.render()
                
                total_rewards.append(episode_reward)
                print(f"Episode {episode + 1}: Steps = {step}, Total Reward = {episode_reward}")
            
            average_reward = np.mean(total_rewards)
            print(f"\nAverage Reward over {num_episodes} episodes: {average_reward:.2f}")
            return average_reward
            
        except Exception as e:
            print(f"Error during testing: {e}")
            return None
        finally:
            if render:
                self.env.close()

if __name__ == "__main__":
    # Test the model
    agent = CMAESAgent("CartPole-v1")
    if agent.load_model("cmaes_model.npy"):
        agent.test_model(num_episodes=5)
    else:
        print("Failed to load model. Exiting.")
