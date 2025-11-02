import torch
import torch.nn as nn
import random
import numpy as np
from collections import deque
import torch.nn.functional as F
from contextlib import nullcontext
import math
import time

class DuelingDQN(nn.Module):
    """Dueling network for improved value estimation and stability"""
    def __init__(self, input_size=24, hidden_size=384, output_size=3):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        h = self.feature(x)
        value = self.value_stream(h)
        advantage = self.advantage_stream(h)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class Agent:
    def __init__(self, state_size=24, action_size=3, lr=0.0008, n_step=5, capacity=75000):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

        # network
        self.q_network = DuelingDQN(state_size, 384, action_size).to(self.device)
        self.target_network = DuelingDQN(state_size, 384, action_size).to(self.device)

        # try to compile the models (PyTorch 2.x)
        try:
            if hasattr(torch, "compile") and self.device.type == "cuda":
                # compile available and running on CUDA -> attempt compile for speedups
                compiled = torch.compile(self._raw_q_network)
                self.q_network = compiled
            else:
                # Skip compile on CPU (common on Windows without cl.exe)
                self.q_network = self._raw_q_network
        except Exception as e:
            # Fallback: keep raw module and warn
            print(f"[warning] torch.compile failed or skipped: {e}")
            self.q_network = self._raw_q_network

        # optimizer / scheduler / scaler unchanged
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.9)
        try:
            self.scaler = torch.amp.GradScaler(device='cuda' if self.device.type == 'cuda' else 'cpu')
        except Exception:
            self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'))

        # n-step
        self.n_step = max(1, int(n_step))
        self.nstep_buffer = deque(maxlen=self.n_step)

        # Use our numpy replay buffer
        self.capacity = capacity
        self.replay_buffer = NumpyReplayBuffer(self.capacity, state_shape=(self.state_size,), dtype=np.float32)

        # prioritized replay params still used in sampling (alpha/beta)
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment = 1e-5
        self.priority_epsilon = 1e-3

        # exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.9992

        # training params
        self.gamma = 0.98
        self.tau = 0.004
        self.target_update_every = 750
        self.steps = 0

        # init target
        self.update_target_network(tau=1.0)

        print(f"Dueling Agent initialized - {sum(p.numel() for p in self.q_network.parameters())} parameters")

    def act(self, state, training=True):
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        # faster tensor creation
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(1).item())

    def remember(self, state, action, reward, next_state, done):
        # n-step logic (kept same)
        self.nstep_buffer.append((state, action, reward, next_state, done))
        if len(self.nstep_buffer) == self.n_step:
            R = 0.0
            next_state_n = self.nstep_buffer[-1][3]
            done_n = False
            gamma_pow = 1.0
            for i, (_, _, r, _, d) in enumerate(self.nstep_buffer):
                R += gamma_pow * r
                gamma_pow *= self.gamma
                if d:
                    done_n = True
                    next_state_n = self.nstep_buffer[i][3]
                    break
            state_0, action_0 = self.nstep_buffer[0][0], self.nstep_buffer[0][1]
            self.replay_buffer.add(state_0, action_0, R, next_state_n, done_n)
        # flush on done
        if done:
            while len(self.nstep_buffer) > 1:
                self.nstep_buffer.popleft()
                R = 0.0
                next_state_n = self.nstep_buffer[-1][3]
                done_n = False
                gamma_pow = 1.0
                for i, (_, _, r, _, d) in enumerate(self.nstep_buffer):
                    R += gamma_pow * r
                    gamma_pow *= self.gamma
                    if d:
                        done_n = True
                        next_state_n = self.nstep_buffer[i][3]
                        break
                state_0, action_0 = self.nstep_buffer[0][0], self.nstep_buffer[0][1]
                self.replay_buffer.add(state_0, action_0, R, next_state_n, done_n)
            self.nstep_buffer.clear()

    def replay(self, batch_size=256):
        if len(self.replay_buffer) < batch_size:
            return

        self.steps += 1

        # sample from numpy buffer
        batch = self.replay_buffer.sample(batch_size=batch_size, alpha=self.alpha)
        indices = batch['indices']
        probs = batch['probs']  # normalized approx probs
        # importance sampling weights
        weights = (len(self.replay_buffer) * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)
        weights_t = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        states = torch.as_tensor(batch['states'], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch['actions'], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch['rewards'], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch['next_states'], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch['dones'], dtype=torch.bool, device=self.device)

        autocast_ctx = torch.cuda.amp.autocast if self.device.type == 'cuda' else nullcontext
        with autocast_ctx():
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                gamma_pows = torch.where(dones, torch.ones_like(next_q_values), torch.full_like(next_q_values, self.gamma ** self.n_step))
                target_q_values = rewards + (gamma_pows * next_q_values)

            td_errors = (target_q_values - current_q_values).detach()
            per_sample_loss = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
            loss = (weights_t * per_sample_loss).mean()

        # backward + step
        self.optimizer.zero_grad(set_to_none=True)
        if self.device.type == 'cuda':
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()

        # update priorities in buffer (vectorized)
        new_prios = (td_errors.abs().cpu().numpy() + self.priority_epsilon).astype(np.float32)
        self.replay_buffer.update_priorities(indices, new_prios)

        # anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        # scheduler step periodically
        if self.steps % 100 == 0:
            self.scheduler.step()

        # soft update
        if self.steps % self.target_update_every == 0:
            self.update_target_network()

        # decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def _store_transition(self, transition):
        max_prio = self.priorities[:len(self.buffer)].max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max(max_prio, self.priority_epsilon)
        self.pos = (self.pos + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)
    
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
            'scheduler': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filename)
    
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.steps = checkpoint.get('steps', 0)

class NumpyReplayBuffer:
    """
    Fixed-size numpy replay buffer with proportional (prefix-sum) sampling.
    Stores states, actions, rewards, next_states, dones and priorities.
    """
    def __init__(self, capacity, state_shape, dtype=np.float32, device='cpu'):
        self.capacity = int(capacity)
        self.state_shape = state_shape
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((self.capacity, *state_shape), dtype=dtype)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=dtype)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.uint8)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        self.min_prio = 1e-6

    def add(self, state, action, reward, next_state, done, prio=None):
        idx = self.ptr
        self.states[idx] = state
        self.next_states[idx] = next_state
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[idx] = 1 if done else 0
        if prio is None:
            # new transitions get max priority
            max_p = self.priorities[:self.size].max() if self.size > 0 else 1.0
            self.priorities[idx] = max(max_p, self.min_prio)
        else:
            self.priorities[idx] = max(prio, self.min_prio)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, alpha=0.6):
        if self.size == 0:
            raise ValueError("Empty buffer")
        # proportional sampling via prefix sum (fast)
        prios = self.priorities[:self.size].astype(np.float64)
        # apply alpha
        probs = prios ** alpha
        total = probs.sum()
        if total == 0:
            probs = np.ones_like(probs) / probs.size
            csum = np.cumsum(probs)
        else:
            csum = np.cumsum(probs)

        # sample uniform numbers and use searchsorted
        random_vals = np.random.rand(batch_size) * csum[-1]
        indices = np.searchsorted(csum, random_vals)
        # clip indices just in case
        indices = np.minimum(indices, self.size - 1)

        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices].astype(np.bool_),
            'indices': indices,
            'probs': probs[indices] / (csum[-1])  # normalized probabilities for IS weights
        }
        return batch

    def update_priorities(self, indices, new_prios):
        # vectorized update
        self.priorities[indices] = np.maximum(new_prios, self.min_prio)

    def __len__(self):
        return self.size