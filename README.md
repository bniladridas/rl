For usage:

```python
from harpertoken.models.model import CMAESAgent

agent = CMAESAgent.from_pretrained("harpertoken/harpertoken-cartpole")

from harpertoken.evaluation.test_model import test_model
test_model(agent, num_episodes=5)
```
