# -----------------------------
# File: DQN Algorithm in CartPole
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.5
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import torch.nn as nn
import torch.nn.functional as F


# 2. Define the network used in both target net and the net for training
class DQNnet(nn.Module):
    def __init__(self, n_states, n_actions):
        # Define the network structure, a very simple fully connected network
        super(DQNnet, self).__init__()
        # Define the structure of fully connected network
        self.fc1 = nn.Linear(n_states, 10)  # layer 1
        self.fc1.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc1
        self.out = nn.Linear(10, n_actions)  # layer 2
        self.out.weight.data.normal_(0, 0.1)  # in-place initilization of weights of fc2

    def forward(self, x):
        # Define how the input data pass inside the network
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)  # 最后得到的是一个二维的数组，两个数分别表示两个动作的值函数
        return actions_value