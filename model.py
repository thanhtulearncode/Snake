import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import torch.nn.functional as F

class DuelingdDQN(nn.Module):
    def __init__(self, input_size=24, hidden_size=256, output_size=3):
        super(DuelingdDQN, self).__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU()
        )
        # Dueling streams
        self.advantage = nn.Linear(hidden_size//2, output_size)
        self.value = nn.Linear(hidden_size//2, 1)
    def forward(self, x):
        features = self.feature_net(x)
        advantage = self.advantage(features)
        value = self.value(features)
        # Dueling aggregation
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values

class Agent:
    def __init__(self, state_size=24, action_size=3, lr=0.0005):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Neural networks
        self.q_network = DuelingdDQN(state_size, 256, action_size).to(self.device)
        self.target_network = DuelingdDQN(state_size, 256, action_size).to(self.device)
        # Optimizer with stable learning rate
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-4)
        # Use step LR instead of cosine annealing for more stability
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.8)
        # Experience replay
        self.memory = deque(maxlen=100000)
        # Exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9995
        # Training parameters
        self.gamma = 0.99
        self.tau = 0.005
        self.update_every = 4
        self.target_update_every = 1000
        self.steps = 0
        # Initialize target network
        self.update_target_network(tau=1.0)
        print(f"Optimized Agent initialized - {sum(p.numel() for p in self.q_network.parameters())} parameters")
    
    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.choice(range(self.action_size))
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        self.steps += 1
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first, then to tensors (fixes the warning)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        # Loss and optimization
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        # Update schedules
        if self.steps % 100 == 0:
            self.scheduler.step()
        # Update target network
        if self.steps % self.target_update_every == 0:
            self.update_target_network()
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self, tau=None):
        if tau is None:
            tau = self.tau
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def save(self, filename):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)
