import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from torch.distributions import Categorical
from tqdm import tqdm
from arc_challenge_solver.utils.loss_functions import ARCLoss  # Added for metric calculation
import nni  # Added for NNI reporting
from arc_challenge_solver.utils.checkpoint import save_checkpoint, load_checkpoint
import torch.nn.functional as F


class PPONetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=4, num_colors=10, dropout_rate=0.1, activation_name='relu', use_skip_connections=True):
        super().__init__()
        self.num_colors = num_colors
        self.use_skip_connections = use_skip_connections
        
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_name == 'elu':
            self.activation = nn.ELU()
        elif activation_name == 'swish':
            self.activation = nn.SiLU()
        
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1))
        
        self.action_layer = nn.Conv2d(hidden_dim, num_colors, kernel_size=1)
        self.value_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.size_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Predict height and width
        )

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.dropout(x)
        
        for layer in self.hidden_layers:
            if self.use_skip_connections:
                residual = x
                x = self.activation(layer(x))
                x = self.dropout(x)
                x = x + residual
            else:
                x = self.activation(layer(x))
                x = self.dropout(x)
        
        action_logits = self.action_layer(x)
        value = self.value_layer(x).squeeze(-1).squeeze(-1)
        size_logits = self.size_predictor(x)
        
        return action_logits, value, size_logits

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        action_logits, value, size_logits = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=1)
        dist = Categorical(action_probs.permute(0, 2, 3, 1).reshape(-1, self.num_colors))
        action = dist.sample().reshape(action_probs.shape[2], action_probs.shape[3])
        log_prob = dist.log_prob(action.reshape(-1)).reshape(action.shape)
        size = torch.clamp(size_logits, min=1, max=30).int().squeeze(0)
        return action.numpy(), log_prob, value.squeeze(), size.numpy()

    def evaluate(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(1)  # Add channel dimension
        action_logits, value, size_logits = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=1)
        dist = Categorical(action_probs.permute(0, 2, 3, 1).reshape(-1, self.num_colors))
        action_flat = action.reshape(-1)
        log_probs = dist.log_prob(action_flat).reshape(action.shape)
        entropy = dist.entropy().reshape(action.shape)
        return log_probs, value.squeeze(-1).squeeze(-1), entropy, torch.clamp(size_logits, min=1, max=30).int()
class PPOModel:
    def __init__(self, train_pairs, hidden_dim=64, num_layers=4, learning_rate=3e-4, batch_size=32, num_epochs=100, dropout_rate=0.1, optimizer_name='adam', activation_name='relu', use_skip_connections=True, weight_decay=1e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_colors = 10  # Assuming 10 color options in ARC
        self.policy = PPONetwork(input_dim=1, hidden_dim=hidden_dim, num_layers=num_layers, num_colors=self.num_colors, dropout_rate=dropout_rate, activation_name=activation_name, use_skip_connections=use_skip_connections).to(device)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'rmsprop':
            self.optimizer = optim.RMSprop(self.policy.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.train_pairs = train_pairs
        self.max_grid_size = 30
        self.training_losses = []
        self.training_rewards = []
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.preprocess_data()
        
    def save_checkpoint(self, epoch: int, score: float, filename: str):
        save_checkpoint(self.policy, self.optimizer, epoch, score, filename)

    def load_checkpoint(self, filename: str):
        checkpoint = load_checkpoint(self.policy, self.optimizer, filename)
        self.best_score = checkpoint['score']
        self.best_epoch = checkpoint['epoch']
        


    def preprocess_data(self):
        self.processed_inputs = []
        self.processed_outputs = []
        for pair in self.train_pairs:
            input_grid = np.array(pair['input'])
            output_grid = np.array(pair['output'])
            
            padded_input = np.zeros((self.max_grid_size, self.max_grid_size))
            padded_input[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
            self.processed_inputs.append(padded_input)
            
            padded_output = np.zeros((self.max_grid_size, self.max_grid_size))
            padded_output[:output_grid.shape[0], :output_grid.shape[1]] = output_grid
            self.processed_outputs.append(padded_output)

        self.processed_inputs = torch.FloatTensor(np.array(self.processed_inputs)).unsqueeze(1).to(self.device)
        self.processed_outputs = torch.LongTensor(np.array(self.processed_outputs)).to(self.device)

    def train(self):
        for epoch in tqdm(range(self.num_epochs), desc="Training PPO"):
            epoch_losses = []
            epoch_rewards = []
            indices = torch.randperm(len(self.processed_inputs))
            for start_idx in range(0, len(indices), self.batch_size):
                batch_indices = indices[start_idx:start_idx+self.batch_size]
                states = self.processed_inputs[batch_indices]
                true_actions = self.processed_outputs[batch_indices]

                action_logits, state_values, size_logits = self.policy(states)
                predicted_size = torch.clamp(size_logits, min=1, max=30).int()
                true_size = torch.tensor([[o.shape[0], o.shape[1]] for o in true_actions]).to(self.device)

                action_probs = torch.softmax(action_logits, dim=1)
                dist = Categorical(action_probs.permute(0, 2, 3, 1).reshape(-1, self.num_colors))
                actions = dist.sample().reshape(action_probs.shape[0], action_probs.shape[2], action_probs.shape[3])

                # Calculate rewards
                rewards = []
                for predicted, true, pred_size, true_size in zip(actions, true_actions, predicted_size, true_size):
                    size_accuracy = 1 - (abs(pred_size[0] - true_size[0]) + abs(pred_size[1] - true_size[1])) / (true_size[0] + true_size[1])
                    pixel_accuracy = (predicted[:true_size[0], :true_size[1]] == true[:true_size[0], :true_size[1]]).float().mean()
                    reward = (size_accuracy + pixel_accuracy) / 2
                    rewards.append(reward)
                rewards = torch.tensor(rewards).to(self.device)

                # Calculate advantages
                advantages = rewards - state_values.squeeze()

                # Calculate log probabilities
                log_probs = dist.log_prob(actions.reshape(-1)).reshape(actions.shape)

                ratio = (log_probs - log_probs.detach()).exp()
                surr1 = ratio * advantages.unsqueeze(-1).unsqueeze(-1)
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages.unsqueeze(-1).unsqueeze(-1)
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = advantages.pow(2).mean()
                entropy = dist.entropy().mean()
                
                size_loss = F.mse_loss(size_logits.float(), true_size.float())
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy + 0.1 * size_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_rewards.append(rewards.mean().item())

            self.training_losses.append(np.mean(epoch_losses))
            self.training_rewards.append(np.mean(epoch_rewards))
            
            nni.report_intermediate_result(np.mean(epoch_rewards))

    def predict(self, input_grid: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        with torch.no_grad():
            padded_input = np.zeros((self.max_grid_size, self.max_grid_size))
            padded_input[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
            state = torch.FloatTensor(padded_input).unsqueeze(0).unsqueeze(0).to(self.device)
            action_logits, _, size_logits = self.policy(state)
            actions = action_logits.argmax(dim=1).squeeze(0)
            predicted_size = torch.clamp(size_logits.squeeze(), min=1, max=30).int()
            h, w = predicted_size.cpu().numpy()
        return actions.cpu().numpy()[:h, :w], (h, w)
    
    def save_checkpoint(self, filepath):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])