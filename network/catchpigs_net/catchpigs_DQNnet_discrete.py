# -----------------------------
# File: DQN Algorithm in Catch Pigs
# Author: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------


import torch.nn as nn
import torch.nn.functional as F


# 神经网络结构，结构较为容易理解
class DeepNetWork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=32):
        super(DeepNetWork, self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob