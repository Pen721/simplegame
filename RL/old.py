import numpy as np
import matplotlib.pyplot as plt

class SimpleResourceGame:
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.resources = 10
        self.step_count = 0
        self.done = False
        self.possible_rewards = []
        return self._get_state()
    
    def get_action_reward_pairs(self):
        raise ValueError("Subclasses must implement this method!")
 
    def step(self, action):
        action_reward_pairs = self.get_action_reward_pairs()
        reward = action_reward_pairs[action]
        self.resources += reward

        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True

        return self._get_state(), reward, self.done, action_reward_pairs

    def _get_state(self):
        return np.array([self.resources, self.step_count])

    def render(self):
        print(f"Step: {self.step_count}, Resources: {self.resources}")

    def get_possible_rewards(self):
        return self.possible_rewards
    
class SpendNotAlwaysGood(SimpleResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)
    
    def get_action_reward_pairs(self):
        # Compute all possible rewards
        action_reward_pairs = {}

        save_reward = 1
        if self.resources > 0:
            spend_rewards = list(range(self.resources + 1))
        else:
            spend_rewards = [-1]  # Penalty for trying to spend when no resources

        # Action: 0 = Save, 1 = Spend
        if self.resources > 0:
            spend_reward = np.random.choice(spend_rewards)
        else:
            spend_reward = -1

        save_reward = save_reward

        action_reward_pairs[0] = save_reward
        action_reward_pairs[1] = spend_reward
        return action_reward_pairs
        
    
class SpendAlwaysGood(SimpleResourceGame):
    def __init__(self, max_steps=100):
        super().__init__(max_steps)
    
    def get_action_reward_pairs(self):
        # Compute all possible rewards
        action_reward_pairs = {}

        save_reward = 1
        spend_rewards = list(range(4))

        spend_reward = np.random.choice(spend_rewards)

        action_reward_pairs[0] = save_reward
        action_reward_pairs[1] = spend_reward
        return action_reward_pairs