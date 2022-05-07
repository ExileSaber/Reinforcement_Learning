# -----------------------------
# File: AC-DPG Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.30
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import torch
import torch.nn as nn


# Actor Net
# Actor：输入是state，输出的是一个确定性的action
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.action_bound = torch.tensor(action_bound)

        # layer
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = a * self.action_bound
        return scaled_a


# Critic Net
# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val