# CartPole CMA-ES Model

This repository contains a CartPole agent trained using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization algorithm.

## Model Description

- **Environment**: CartPole-v1 from Gymnasium
- **Algorithm**: CMA-ES (Evolutionary Strategy)
- **Architecture**: Linear policy (state -> action)
- **State Space**: 4 dimensions (cart position, cart velocity, pole angle, pole angular velocity)
- **Action Space**: 2 actions (move left or right)

## Usage

```python
from huggingface_hub import hf_hub_download
from test_model import CMAESAgent

# Initialize agent
agent = CMAESAgent.from_pretrained("bniladridas/cartpole-cmaes")

# Test the model
agent.test_model(num_episodes=5)
```

## Training Details

- Optimization: CMA-ES
- Episodes: 100
- Fitness: Average reward over episodes
- Selection: Best performing solutions used to update search distribution

## Performance

The model achieves an average reward of X over Y episodes in the CartPole-v1 environment.