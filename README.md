For usage:

```python
from harpertoken.models.model import CMAESAgent

# Load pretrained model
agent = CMAESAgent.from_pretrained("harpertoken/harpertoken-cartpole")

# Use for inference
action = agent.get_action(state)
# or evaluate performance
mean_reward, std_reward = agent.evaluate(num_episodes=10)
```

For testing the model:

```python
from harpertoken.evaluation.test_model import test_model
test_model(agent, num_episodes=5)
```
