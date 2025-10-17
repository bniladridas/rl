import pytest
from src.evaluation.test_model import test_model
from src.models.model import CMAESAgent
import numpy as np

def test_test_model(capsys):
    agent = CMAESAgent()
    agent.weights = np.random.rand(8)
    test_model(agent, num_episodes=1)
    captured = capsys.readouterr()
    assert "Mean reward" in captured.out