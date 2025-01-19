import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

class Policy(nn.Module):
    def __init__(self):
        super().__init__()

class OneStatePolicy(Policy):
    def __init__(self):
        super().__init__()

    def visualize_states(self):
        landscape = {}
        for resource in range(-10, 50):
            for step in range(0, 100):
                state = torch.FloatTensor([resource, step])
                chance_of_spend = self.forward(state)[1].item()
                landscape[(resource, step)] = chance_of_spend
        return landscape
    
    def plot_policy_heatmap(self):
        landscape = self.visualize_states()
        
        # Convert the dictionary to numpy arrays
        resources, steps, chances = zip(*[(r, s, c) for (r, s), c in landscape.items()])

        # Create a 2D grid
        resource_unique = sorted(set(resources))
        step_unique = sorted(set(steps))
        chances_grid = np.zeros((len(resource_unique), len(step_unique)))
        
        for (r, s), c in landscape.items():
            i = resource_unique.index(r)
            j = step_unique.index(s)
            chances_grid[i, j] = c
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.imshow(chances_grid, cmap='coolwarm', aspect='auto', origin='lower')
        plt.colorbar(label='Chance of Gambling')
        plt.title('Policy Behavior: Chance of Gambling across States (Resources, EpisodeStep)')
        plt.xlabel('EpisodeStep')
        plt.ylabel('Resources')
        
        # Set tick labels
        step_ticks = np.linspace(0, len(step_unique)-1, 6).astype(int)
        plt.xticks(step_ticks, [step_unique[i] for i in step_ticks])
        
        resource_ticks = np.linspace(0, len(resource_unique)-1, 6).astype(int)
        plt.yticks(resource_ticks, [resource_unique[i] for i in resource_ticks])
        
        plt.show()

class allPastStatePolicy(Policy):
    def __init__(self):
        super().__init__()

    def visualize_states(self):
        return "Not Implemented Yet!"
    
    def plot_policy_heatmap(self):
        return "Not Implemented Yet!"
