import torch
import numpy as np
import matplotlib.pyplot as plt
from policies.Policy import allPastStatePolicy

def sample_trajectories(policy, game, n=10, max_step=10):
    trajectories = []

    for _ in range(n):
        trajectory = []
        env = game() 
        state = env.reset()
        done = False
        step = 0
        while (not done) and (step < max_step):
            actions = env.actions
            states = env.states
            print("actions and states")
            print(actions)
            print(states)
            action_probs = policy.get_next_action_dist(states, actions)

            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()

            result = env.step(action)

            next_state, reward, done = result['state'], result['reward'], result['done']

            trajectory.append((state, action, reward))

            state = next_state
            step += 1
        trajectories.append(trajectory)
    return trajectories

def get_trajectories_mean(trajectories):
    resources = [[step[0][0] for step in trajectory] for trajectory in trajectories]
    return np.mean(np.array(resources)), np.std(np.array(resources))

def plot_rewards(trajectories):
    time = np.arange(len(trajectories[0]))
    n = len(trajectories)

    resources = [[step[0][0] for step in trajectory] for trajectory in trajectories]
    actions = [[step[1] for step in trajectory] for trajectory in trajectories]
    
    plt.figure(figsize=(10, 6))
    for resource in resources:
        plt.plot(time, resource, alpha=0.1, color='#1f77b4')

    for resource, action in zip(resources, actions):
        for i, (x, y, a) in enumerate(zip(time, resource, action)):
            if a == 1:
                plt.plot(x, y, 's', markersize=5, color='red', alpha = 0.1)
            if a == 0:
                plt.plot(x, y, '-o', markersize=5, alpha= 0.1, color='#1f77b4')

    plt.xlabel('Time')
    plt.ylabel('Resources')
    plt.title('Resource Over Time')
    plt.grid(True)
    plt.show()

    # Print final state
    # print(f"Final state - Time: {time[-1]}, Resources: {[r[-1] for r in resources]}")
    print(f"Average Reward at time by time {time[-1]}, {len(trajectories)} sampled trajectories: {np.mean([r[-1] for r in resources])}")

def prepare_sequence(history, max_length=50):
    """
    Prepare a sequence of (state, action) tuples for the transformer.
    States remain as raw values, actions as binary indices.
    """
    states, actions = zip(*history)
    
    # Convert to numpy arrays
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    
    # Pad or truncate sequence
    seq_len = len(states)
    if seq_len > max_length:
        # Take the most recent max_length elements
        states = states[-max_length:]
        actions = actions[-max_length:]
    elif seq_len < max_length:
        # Pad with zeros
        pad_len = max_length - seq_len
        states = np.pad(states, (pad_len, 0))
        actions = np.pad(actions, (pad_len, 0))
    
    return torch.FloatTensor(states), torch.FloatTensor(actions)