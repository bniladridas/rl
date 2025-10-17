import pytest
import numpy as np
from src.models.model import CMAESAgent

def test_cmaes_agent_init():
    agent = CMAESAgent()
    assert agent.observation_space == 4
    assert agent.action_space == 2
    assert agent.weights is None

def test_get_action():
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)  # 4*2
    state = np.random.rand(4)
    action = agent.get_action(state)
    assert action in [0, 1]

def test_evaluate():
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)
    mean_reward, std_reward = agent.evaluate(num_episodes=2)
    assert isinstance(mean_reward, float)
    assert isinstance(std_reward, float)
    assert mean_reward >= 0

def test_save_pretrained(tmp_path):
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)
    save_dir = tmp_path / "model"
    agent._save_pretrained(save_dir)
    assert (save_dir / "model.npy").exists()

def test_from_pretrained(tmp_path):
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)
    save_dir = tmp_path / "model"
    agent._save_pretrained(save_dir)
    
    loaded_agent = CMAESAgent._from_pretrained(model_id=str(save_dir))
    np.testing.assert_array_equal(agent.weights, loaded_agent.weights)
    assert agent.observation_space == loaded_agent.observation_space
    assert agent.action_space == loaded_agent.action_space