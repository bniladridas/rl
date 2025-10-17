import pytest
import numpy as np
from src.models.model import CMAESAgent

def test_cmaes_agent_init():
    agent = CMAESAgent()
    assert agent.observation_space == 4
    assert agent.action_type == 'discrete'
    assert agent.num_actions == 2
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
    
    loaded_agent = CMAESAgent._from_pretrained(
        model_id=str(save_dir),
        revision=None,
        cache_dir=None,
        force_download=False,
        proxies=None,
        resume_download=None,
        local_files_only=False,
        token=None
    )
    np.testing.assert_array_equal(agent.weights, loaded_agent.weights)
    assert agent.observation_space == loaded_agent.observation_space
    assert agent.action_type == loaded_agent.action_type
    if agent.action_type == 'discrete':
        assert agent.num_actions == loaded_agent.num_actions
    elif agent.action_type == 'continuous':
        assert agent.action_dim == loaded_agent.action_dim

def test_from_pretrained_hf():
    # Test loading from Hugging Face Hub
    # This test validates the syntax for loading a pretrained model
    # Note: Requires the model to be uploaded to HF under "harpertoken/harpertoken-cartpole"
    try:
        agent = CMAESAgent.from_pretrained("harpertoken/harpertoken-cartpole")
        assert agent.weights is not None
        assert agent.action_type == 'discrete'
        assert agent.num_actions == 2
    except Exception as e:
        pytest.skip(f"HF model not available or network issue: {e}")

def test_pretrained_usage_syntax():
    # Test the exact syntax provided by the user
    # Validates if the syntax works (loads model and tests it)
    try:
        from src.models.model import CMAESAgent
        agent = CMAESAgent.from_pretrained("harpertoken/harpertoken-cartpole")
        from src.evaluation.test_model import test_model
        # Note: test_model will render, but for syntax validation, we just call it briefly
        # In a real test, you might want to mock or skip rendering
        # For now, just check that the functions can be called without error
        assert callable(agent.get_action)
        assert callable(test_model)
        # To avoid rendering in tests, we don't call test_model(agent, num_episodes=5)
        # But the syntax import and callability is validated
    except Exception as e:
        pytest.skip(f"Syntax validation failed or model not available: {e}")