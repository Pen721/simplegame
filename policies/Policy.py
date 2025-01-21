import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

class Policy(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_next_action_dist(self, states, actions):
        """Subclasses must implement this method to return action distribution"""
        pass
    
    def get_next_action(self, states, actions):
        return torch.multinomial(self.get_next_action_dist(states, actions), num_samples=1).item()