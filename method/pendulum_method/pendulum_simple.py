# -----------------------------
# File: AC-Simple Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.29
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../network/pendulum_net/")
from pendulum_ACnet_SPG_continue import ActorNet, CriticNet

import torch
import torch.nn as nn
import numpy as np


class Simple(nn.Module):
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.device = args.device
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.a_update_steps = args.a_update_steps
        self.c_update_steps = args.c_update_steps

        self._build()

    def _build(self):
        self.actor_model = ActorNet(self.n_states, self.bound).to(self.device)
        self.actor_old_model = ActorNet(self.n_states, self.bound).to(self.device)
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr_a)

        self.critic_model = CriticNet(self.n_states).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample().cpu()
        return np.clip(action, -self.bound, self.bound)

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_).to(self.device)
        target = self.critic_model(s_).detach()  # torch.Size([1])
        target_list = []
        for r in rewards[::-1]:
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()
        target_list = torch.cat(target_list)  # torch.Size([batch])

        return target_list

    def actor_learn(self, states, actions, advantage):
        # print(states)
        # print(actions)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))
        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 1])
        loss = -torch.mean(surr)

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states).to(self.device)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states).to(self.device)
        v = self.critic_model(states)  # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)

        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)
