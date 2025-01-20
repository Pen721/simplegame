import numpy as np

class SimpleResourceGame:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.resources = 10
        self.step_count = 0
        self.done = False
        return self._get_state()

    def step(self, action):
        # Action: 0 = Save, 1 = Spend
        if action == 1:  # Spend
            if self.resources > 0:
                reward = np.random.randint(0, self.resources + 1)
                self.resources -= 1
            else:
                reward = -1  # Penalty for trying to spend when no resources
        else:  # Save
            self.resources += 1
            reward = 0
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        return np.array([self.resources, self.step_count])

    def render(self):
        print(f"Step: {self.step_count}, Resources: {self.resources}")