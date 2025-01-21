import torch
import matplotlib.pyplot as plt

def discount_rewards(trajectory, gamma=0.9):
    rewards = [t[2] for t in trajectory]
    discounted_rewards = torch.zeros_like(torch.tensor(rewards, dtype=torch.float32))
    R = 0
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R
        discounted_rewards[t] = R
    return discounted_rewards

def get_loss_for_batch(trajectories, action_logs):
    losses = []
    all_rewards = []
    
    # First collect all discounted rewards
    for trajectory, action_log in zip(trajectories, action_logs):
        discounted_rewards = discount_rewards(trajectory)
        all_rewards.append(discounted_rewards)
    
    # Concatenate and normalize across all trajectories
    all_rewards = torch.cat(all_rewards)
    normalized_rewards = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-8)
    
    # Split back into trajectory-sized chunks
    start_idx = 0
    for trajectory, action_log in zip(trajectories, action_logs):
        length = min(len(action_log), len(trajectory))
        trajectory_rewards = normalized_rewards[start_idx:start_idx + length]
        episode_loss = -1 * (action_log[:length] * trajectory_rewards).sum()
        losses.append(episode_loss)
        start_idx += length

    return torch.stack(losses).mean()

def sample_trajectories(policy, game, n=10, max_step=100):
    trajectories = []
    action_logs = []

    for _ in range(n):
        trajectory = []
        env = game()  # Assuming game() is defined elsewhere
        state = env.reset()
        episode_action_logs = []
        actions = []

        done = False
        while not done:
            # Original behavior for non-transformer policies
            states = env.states
    
            if len(actions) == 0:
                actions = [-1]
            action_probs = policy.get_next_action_dist(states, actions)

            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            
            log_prob = action_distribution.log_prob(action)
            actions.append(action.item())

            result = env.step(action.item())
            next_state, reward, done = result['state'], result['reward'], result['done']

            trajectory.append((state, action.item(), reward))
            episode_action_logs.append(log_prob)
            state = next_state

        trajectories.append(trajectory)
        action_logs.append(torch.stack(episode_action_logs))
    return trajectories, action_logs
    
    
def reinforce(policy, game, optimizer, n=10, max_step=100):
    trajectories = []
    action_logs = []

    trajectories, action_logs = sample_trajectories(policy, game, n=n, max_step=max_step)

    loss = get_loss_for_batch(trajectories, action_logs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def plot_loss(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Loss over time')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()