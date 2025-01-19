import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
from policies.Policy import Policy
from policies.Policy import OneStatePolicy, allPastStatePolicy

class OneFCPolicy(OneStatePolicy):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)
    
class TwoFCPolicy(OneStatePolicy):
    def __init__(self, input_dim=2, embedding_dim=8, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc(x))), dim=-1)
    

class ThreeFCPolicy(OneStatePolicy):
    def __init__(self, input_dim=2, embedding_dim=8, output_dim=2):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
        self.fc2 = nn.Linear(embedding_dim, embedding_dim)
        self.fc3 = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)
    
class DefineFCLayersPolicy(OneStatePolicy):
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
    
class TransformerPolicy(Policy):
    def __init__(self, input_dim=2, d_model=8, nhead=4, num_layers=4, output_dim=2, context_length=100):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, context_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.positional_encoding[:, :x.size(1), :]
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)  # (batch_size, d_model)
        x = self.fc_out(x)  # (batch_size, output_dim)
        return torch.softmax(x, dim=-1)
    
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


    
    
