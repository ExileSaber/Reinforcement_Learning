# -----------------------------
# File: AC-discrete Algorithm in Olympics Curling
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(Actor, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64):
        super(Critic, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value
