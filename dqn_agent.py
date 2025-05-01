import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from typing import Tuple, List, Deque, Any
from pathlib import Path


class DQN(nn.Module):
    def __init__(self, input_size: int = 20, output_size: int = 4):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),  # ↑ increased from 128 for better learning
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QTrainer:
    def __init__(self, model: DQN, lr: float = 0.001, gamma: float = 0.9):
        self.model = model
        self.gamma = 0.99
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Slower learning rate
        self.criterion = nn.SmoothL1Loss()  # Better for Q-learning than MSELoss
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.bool).to(self.device)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = done.unsqueeze(0)

        pred = self.model(state)
        target = pred.clone()
        
        with torch.no_grad():
            next_pred = self.model(next_state)
            max_next_q = next_pred.max(1)[0]

        for i in range(len(done)):
            target[i][action[i]] = reward[i] if done[i] else reward[i] + self.gamma * max_next_q[i]

        self.optimizer.zero_grad()
        loss = self.criterion(pred, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class DQNAgent:
    def __init__(self, input_size: int = 11, output_size: int = 4):
        self.model = DQN(input_size=20, output_size=4)
        self.memory: Deque[Tuple] = deque(maxlen=100_000)
        self.batch_size = 1000
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995  # already good, you can slow it to 0.997 for longer exploration
        self.trainer = QTrainer(self.model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def get_action(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path: str):
        Path(path).parent.mkdir(exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.model.eval()