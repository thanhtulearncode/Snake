import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, output_size=3):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:
    def __init__(self, state_size=15, action_size=3, lr=0.00005, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.999
        self.gamma = gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Neural networks
        self.q_network = DQN(state_size, 256, action_size).to(self.device)
        self.target_network = DQN(state_size, 256, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        # Copy weights to target network
        self.update_target_network()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        # Double DQN: use online network to select actions, target network to evaluate
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
    
    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target_network()