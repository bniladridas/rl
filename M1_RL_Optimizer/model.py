import numpy as np
import gymnasium as gym
from huggingface_hub import hf_hub_download
from huggingface_hub import ModelHubMixin
import json
import os
from pathlib import Path
from typing import Optional, Dict, Union

class CMAESAgent(ModelHubMixin):
    def __init__(self, env_name="CartPole-v1"):
        super().__init__()
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.weights = None

    def get_action(self, state):
        weights_matrix = self.weights.reshape(self.observation_space, self.action_space)
        action_scores = np.dot(state, weights_matrix)
        return int(np.argmax(action_scores))

    def evaluate(self, num_episodes=100, render=False):
        total_rewards = []
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.get_action(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        return np.mean(total_rewards), np.std(total_rewards)

    def _save_pretrained(self, save_directory: Union[str, Path]) -> None:
        """Save model weights"""
        os.makedirs(save_directory, exist_ok=True)
        save_path = os.path.join(save_directory, "model.npy")
        np.save(save_path, {
            'weights': self.weights,
            'observation_space': self.observation_space,
            'action_space': self.action_space
        })

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Optional[Union[str, bool]],
        **model_kwargs,
    ) -> "CMAESAgent":
        """Load model weights"""
        # Initialize the model
        model = cls(**model_kwargs)
        
        # Determine the model path
        if os.path.isdir(model_id):
            model_path = os.path.join(model_id, "model.npy")
        else:
            model_path = hf_hub_download(
                repo_id=model_id,
                filename="model.npy",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                token=token
            )
            
        # Load the model data
        model_data = np.load(model_path, allow_pickle=True).item()
        model.weights = model_data['weights']
        model.observation_space = model_data['observation_space']
        model.action_space = model_data['action_space']
        
        return model
