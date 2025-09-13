import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_size=18, hidden_size=512, output_size=3):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, output_size)
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, 1)
        )
    
    def forward(self, x):
        x = self.feature(x)
        advantage = self.advantage(x)
        value = self.value(x)
        return value + advantage - advantage.mean()

class Agent:
    def __init__(self, state_size=18, action_size=3, lr=0.0002, gamma=0.99, tau=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200000)
        self.epsilon = 1.0
        self.epsilon_min = 0.005  
        self.epsilon_decay = 0.9995
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Neural networks
        self.q_network = DuelingDQN(state_size, 512, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, 512, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.loss_fn = nn.SmoothL1Loss()
        # Copy weights to target network
        self.update_target_network(tau=1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.9)
        self.priorities = deque(maxlen=200000)
    
    def remember(self, state, action, reward, next_state, done, priority=1.0):
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=256):  
        if len(self.memory) < batch_size:
            return
        # Simple random sampling instead of prioritized
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        # Double DQN
        next_actions = self.q_network(next_states).max(1)[1].detach()
        next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        # Calculate loss
        loss = self.loss_fn(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 5.0)
        self.optimizer.step()
        self.lr_scheduler.step()
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Soft update target network
        self.update_target_network(self.tau)
    
    def update_target_network(self, tau=0.005):
        """Soft update of the target network parameters"""
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        torch.save(self.q_network.state_dict(), filename)
    
    def load(self, filename):
        self.q_network.load_state_dict(torch.load(filename, map_location=self.device))
        self.update_target_network(tau=1.0)