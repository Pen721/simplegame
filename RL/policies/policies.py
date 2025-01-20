import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from policies.Policy import Policy
from policies.Policy import OneStatePolicy, allPastStatePolicy
import torch
import torch.nn as nn
import numpy as np

class SingleStatePolicy(Policy):
    def __init__(self):
        super().__init__()

    def get_next_action(self, state):
        return self.forward(state)
    
    def get_next_action_dist(self, states, actions):
        return self.forward(torch.FloatTensor(states[-1]))

class OneFCPolicy(SingleStatePolicy):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
    
class TwoFCPolicy(SingleStatePolicy):
    def __init__(self, input_dim=2, embedding_dim=8, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc(x))), dim=-1)
    
class ThreeFCPolicy(SingleStatePolicy):
    def __init__(self, input_dim=2, embedding_dim=8, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    
class DefineFCLayersPolicy(SingleStatePolicy):
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=2, num_layers=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return torch.softmax(x, dim=-1)
    

class AllPastStatePolicy(Policy):
    def __init__(self):
        super().__init__()
    
    def get_next_action_dist(self, states, actions):
        # """
        # Prepare a sequence of (state, action, reward) tuples for the transformer.
        # States and rewards remain as raw values, actions as binary indices.
        # """
        # Add batch dimension
        states = states if torch.is_tensor(states) else torch.tensor(states)
        actions = actions if torch.is_tensor(actions) else torch.tensor(actions)
        if states.dim() == 2:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
            
        # Get action probabilities
        return self.forward(states, actions)
    
    def get_next_action(self, states, actions):
        return torch.multinomial(self.get_next_action_dist(states, actions), num_samples=1).item()

class TransformerRLPolicy(AllPastStatePolicy):
    def __init__(
        self,
        state_dim=2,  # (resource, timestep)
        d_model=64,
        nhead=4,
        num_layers=2,
        max_seq_length=50
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Process state vector
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Action embedding for binary action
        self.action_embedding = nn.Embedding(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # Project the concatenated features
        self.project = nn.Linear(d_model * 2, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head for binary policy
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
    def forward(self, states, actions):
        # states: (batch_size, seq_len, 2) - [resource, timestep]
        # actions: (batch_size, seq_len) - binary indices (0 or 1)
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Process state vectors
        states_reshaped = states.view(-1, states.shape[-1])
        state_features = self.state_net(states_reshaped)
        state_features = state_features.view(batch_size, seq_len, -1)
        
        # Embed actions
        actions = actions.long()
        action_emb = self.action_embedding(actions)
        
        # Combine features
        combined = torch.cat([state_features, action_emb], dim=-1)
        combined = self.project(combined)
        
        # Add positional embeddings
        combined = combined + self.pos_embedding[:, :seq_len, :]
        
        # Attention mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.to(states.device)
        
        # Transformer and policy head
        transformer_out = self.transformer(combined, mask=mask)
        last_hidden = transformer_out[:, -1]
        action_logits = self.policy_head(last_hidden)
        
        return torch.softmax(action_logits, dim=-1)

    
# class betterTransformerPolicy(TransformerPolicy):
#     def __init__(self, resource_vocab_size, step_vocab_size, action_vocab_size, d_model=12, nhead=4, num_layers=4, max_seq_length=100):
#         super().__init__()

#     def forward(self, resources, steps, actions):
#         # Embed resources, steps, and actions
#         resource_embed = self.resource_embedding(resources)
#         step_embed = self.step_embedding(steps)
#         action_embed = self.action_embedding(actions)
        
#         # Combine embeddings
#         x = torch.cat([resource_embed, step_embed, action_embed], dim=-1)
        
#         # Add positional encoding
#         x = x + self.positional_encoding[:, :x.size(1), :]
        
#         # Transformer expects shape: (seq_len, batch_size, d_model)
#         x = x.permute(1, 0, 2)
        
#         # Pass through Transformer
#         transformer_output = self.transformer_encoder(x)
        
#         # Output layer for each step in the sequence
#         outputs = self.fc_out(transformer_output)  # shape: (seq_len, batch_size, action_dim)
        
#         # Apply softmax to get action probabilities for each step
#         action_probs = torch.softmax(outputs, dim=-1)
        
#         return action_probs.permute(1, 0, 2)  # Return shape: (batch_size, seq_len, action_dim)

def compute_subsequence_loss(action_probs, actions, rewards):
    losses = []
    for t in range(action_probs.size(1)):
        # Compute loss for subsequence up to time t
        sub_probs = action_probs[:, :t+1, :]
        sub_actions = actions[:, :t+1]
        sub_rewards = rewards[:, :t+1]
        
        log_probs = torch.log(sub_probs.gather(2, sub_actions.unsqueeze(-1)).squeeze(-1))
        discounted_rewards = compute_discounted_rewards(sub_rewards)
        
        loss = -(log_probs * discounted_rewards).sum()
        losses.append(loss)
    
    return torch.stack(losses).mean()


    
    
