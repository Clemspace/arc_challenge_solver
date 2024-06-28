import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm



class PPONetwork(nn.Module):
    def __init__(self, input_dim=900, hidden_dim=256, num_colors=10):
        super().__init__()
        self.num_colors = num_colors
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, num_colors, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        self.critic = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # x shape: (batch_size, 1, 30, 30)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        action_logits = self.conv3(x)  # shape: (batch_size, num_colors, 30, 30)
        value = self.critic(x).squeeze(1)  # shape: (batch_size, 30, 30)
        
        return action_logits, value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze()

    def evaluate(self, state, action):
        state = torch.FloatTensor(state)
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_log_probs, torch.squeeze(value), dist_entropy

class PPOModel:
    def __init__(self, train_pairs: List[Dict], device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_colors = 10  # Assuming 10 color options in ARC
        self.policy = PPONetwork(num_colors=self.num_colors).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.train_pairs = train_pairs
        self.max_grid_size = 30
        self.training_losses = []
        self.training_rewards = []
        self.preprocess_data()

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

    def train(self, num_epochs=100, batch_size=64):
        for epoch in tqdm(range(num_epochs), desc="Training PPO"):
            epoch_losses = []
            epoch_rewards = []
            indices = torch.randperm(len(self.processed_inputs))
            for start_idx in range(0, len(indices), batch_size):
                batch_indices = indices[start_idx:start_idx+batch_size]
                states = self.processed_inputs[batch_indices]
                true_actions = self.processed_outputs[batch_indices]

                action_logits, state_values = self.policy(states)
                action_probs = torch.softmax(action_logits, dim=1)
                dist = Categorical(action_probs.permute(0, 2, 3, 1))
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                advantages = (true_actions.float() - state_values)

                ratio = (log_probs - log_probs.detach()).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = advantages.pow(2).mean()
                entropy = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                epoch_rewards.append(-critic_loss.item())

            self.training_losses.append(np.mean(epoch_losses))
            self.training_rewards.append(np.mean(epoch_rewards))

    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            padded_input = np.zeros((self.max_grid_size, self.max_grid_size))
            padded_input[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
            state = torch.FloatTensor(padded_input).unsqueeze(0).unsqueeze(0).to(self.device)
            action_logits, _ = self.policy(state)
            actions = action_logits.argmax(dim=1).squeeze(0)
        return actions.cpu().numpy()[:input_grid.shape[0], :input_grid.shape[1]]


    
    def update_policy(self, states, actions, rewards, dones, old_log_probs, values, clip_param=0.2):
        returns = self.compute_returns(rewards, dones)
        advantages = returns - values
        
        total_loss = 0
        for _ in range(4):  # 4 policy update iterations
            action_probs, new_values = self.policy(torch.stack(states))
            action_dist = Categorical(logits=action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy = action_dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / 4
        avg_reward = rewards.mean().item()
        
        return avg_loss, avg_reward



    def collect_batch(self, batch_size):
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
        
        for _ in range(batch_size):
            state, _ = self.env.reset()
            done = False
            
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action_probs, value = self.policy(state_tensor)
                action_dist = Categorical(logits=action_probs)
                action = action_dist.sample()
                
                next_state, reward, done, _, _ = self.env.step(action.item())
                
                states.append(state)
                actions.append(action.item())
                rewards.append(reward)
                dones.append(done)
                log_probs.append(action_dist.log_prob(action))
                values.append(value)
                
                state = next_state
        
        return (
            torch.FloatTensor(np.array(states)),  # Convert to numpy array first
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.BoolTensor(dones),
            torch.stack(log_probs),
            torch.cat(values)
        )

    def compute_returns(self, rewards, dones, gamma=0.99):
        returns = torch.zeros_like(rewards)
        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * (1 - dones[t])
            returns[t] = running_return
        return returns

    def update_policy(self, states, actions, rewards, dones, old_log_probs, values, clip_param=0.2):
        returns = self.compute_returns(rewards, dones)
        advantages = (returns - values).detach()
        
        for _ in range(4):  # 4 policy update iterations
            action_probs, new_values = self.policy(states)
            action_dist = Categorical(logits=action_probs)
            new_log_probs = action_dist.log_prob(actions)
            
            ratio = (new_log_probs - old_log_probs.detach()).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            entropy = action_dist.entropy().mean()
            
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
        
        return loss.item(), rewards.mean().item()

    def predict(self, input_grid: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            padded_input = np.zeros((30, 30), dtype=np.int32)
            padded_input[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
            state_tensor = torch.FloatTensor(padded_input).unsqueeze(0)
            action_probs, _ = self.policy(state_tensor)
            action = action_probs.argmax().item()
        return np.full_like(input_grid, action)
