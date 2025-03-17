---
language: en
tags:
- reinforcement-learning
- gymnasium
- cartpole
- cma-es
license: mit
library_name: custom
datasets:
- gymnasium/CartPole-v1
metrics:
- mean_reward
model-index:
- name: CartPole-CMA-ES
  results:
  - task: 
      type: reinforcement-learning
      name: CartPole-v1
    dataset:
      name: gymnasium/CartPole-v1
      type: gymnasium
    metrics:
      - type: mean_reward
        value: 500.00
        name: Mean Reward
---

# CartPole CMA-ES Agent

This model implements a CartPole agent trained using the CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.

## Model Description

- **Model Type:** Reinforcement Learning Policy
- **Training Algorithm:** CMA-ES
- **Environment:** CartPole-v1 (Gymnasium)
- **Input:** State vector (4 dimensions)
- **Output:** Discrete action (2 dimensions)
- **Last Updated:** 2025-03-17

## Performance Metrics

- **Mean Reward:** 500.00 ± 0.00
- **Evaluation Episodes:** 10
- **Training Episodes:** 100 generations with population size 16

## Usage

```python
from model import CMAESAgent

# Load the model
agent = CMAESAgent.from_pretrained("bniladridas/cartpole-cmaes")

# Evaluate
mean_reward, std_reward = agent.evaluate(num_episodes=5)
print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
```

## Training Details

The agent was trained using the CMA-ES algorithm with the following specifications:
- Linear policy mapping states directly to actions
- No neural networks involved - pure evolutionary optimization
- Population size: 16
- Generations: 100

## Limitations and Biases

- The model is specifically trained for the CartPole-v1 environment
- Performance may vary due to the stochastic nature of the environment
- The linear policy might not generalize well to more complex tasks

## Citation

```bibtex
@misc{cartpole-cmaes,
  author = {Niladri Das},
  title = {CartPole CMA-ES Agent},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Hub},
  howpublished = {\url{https://huggingface.co/bniladridas/cartpole-cmaes}}
}
```