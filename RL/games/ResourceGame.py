import numpy as np
import matplotlib.pyplot as plt

class ResourceGame:
    """
    Simple resource game, agent decides to gamble or save at each step. 
    """
    def __init__(self, max_steps=100):
        self.max_steps = max_steps
        self.reset()
        self.actions = []
        self.states = []

    def reset(self):
        self.resources = 10
        self.step_count = 0
        
        self.done = False
        self.actions = []
        self.states = []

        self.states.append(self._get_state())
        return self._get_state()
    
    def update_states(self, action):
        assert(action in [0, 1])

    def step(self, action):
        reward = self.update_states(action)
        self.actions.append(action)
        self.states.append(self._get_state())
        return {'state' : self._get_state(), 'reward' : reward, 'done' : self.done, 'other' : {}}

    def _get_state(self):
        return np.array([self.resources, self.step_count])

    def render(self):
        print(f"Step: {self.step_count}, Resources: {self.resources}")