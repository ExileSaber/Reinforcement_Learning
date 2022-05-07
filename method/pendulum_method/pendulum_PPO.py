# -----------------------------
# File: AC-PPO Algorithm in Pendulum
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


# PPO主要的算法实现部分
class PPO(nn.Module):
    def __init__(self, n_states, n_actions, bound, args):
        super().__init__()
        self.n_states = n_states
        self.n_actions = n_actions
        self.bound = bound
        self.device = args.device
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.gamma = args.gamma
        self.epsilon = args.epsilon  # PPO的clipping方法裁剪的范围
        self.a_update_steps = args.a_update_steps  # actor网络每次调用更新函数的更新次数
        self.c_update_steps = args.c_update_steps  # critic网络每次调用更新函数的更新次数

        self._build()  # 初始化网络部分

    def _build(self):
        self.actor_model = ActorNet(self.n_states, self.bound).to(self.device)
        self.actor_old_model = ActorNet(self.n_states, self.bound).to(self.device)  # 两个actor网络,计算TD目标时选择下一个动作的网络
        self.actor_optim = torch.optim.Adam(self.actor_model.parameters(), lr=self.lr_a)

        self.critic_model = CriticNet(self.n_states).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic_model.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        s = torch.FloatTensor(s).to(self.device)
        mu, sigma = self.actor_model(s)
        dist = torch.distributions.Normal(mu, sigma)  # 生成特定的分布
        action = dist.sample().cpu()  # 根据概率分布采样生成action动作
        return np.clip(action, -self.bound, self.bound)  # 调整动作区间

    def discount_reward(self, rewards, s_):
        s_ = torch.FloatTensor(s_).to(self.device)
        target = self.critic_model(s_).detach()  # torch.Size([1]),计算的是s_状态下的真实value function
        target_list = []
        for r in rewards[::-1]:  # 通过完整路径计算了每个step的真实value function
            target = r + self.gamma * target
            target_list.append(target)
        target_list.reverse()  # 需要反向,使得state和真实value function对应
        target_list = torch.cat(target_list)  # torch.Size([batch]),将多个值拼接到一起,每一个都是一个torch.tensor

        return target_list

    def actor_learn(self, states, actions, advantage):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).reshape(-1, 1).to(self.device)

        mu, sigma = self.actor_model(states)
        pi = torch.distributions.Normal(mu, sigma)

        old_mu, old_sigma = self.actor_old_model(states)
        old_pi = torch.distributions.Normal(old_mu, old_sigma)

        ratio = torch.exp(pi.log_prob(actions) - old_pi.log_prob(actions))  # log_prob(value)是计算value在定义的正态分布（mean,1）中对应的概率的对数
        surr = ratio * advantage.reshape(-1, 1)  # torch.Size([batch, 1])
        loss = -torch.mean(
            torch.min(surr, torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantage.reshape(-1, 1)))  # clipping 方法

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def critic_learn(self, states, targets):
        states = torch.FloatTensor(states).to(self.device)
        v = self.critic_model(states).reshape(1, -1).squeeze(0)  # 预测的value function

        loss_func = nn.MSELoss()
        loss = loss_func(v, targets)  # targets是真实的value function

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    # 计算真实value function和预测的差值
    def cal_adv(self, states, targets):
        states = torch.FloatTensor(states).to(self.device)
        v = self.critic_model(states)  # torch.Size([batch, 1])
        advantage = targets - v.reshape(1, -1).squeeze(0)  # 计算真实value function和预测的差值
        return advantage.detach()  # torch.Size([batch])

    def update(self, states, actions, targets):
        self.actor_old_model.load_state_dict(self.actor_model.state_dict())  # 首先更新旧模型
        advantage = self.cal_adv(states, targets)

        for i in range(self.a_update_steps):  # 更新多次
            self.actor_learn(states, actions, advantage)

        for i in range(self.c_update_steps):  # 更新多次
            self.critic_learn(states, targets)
