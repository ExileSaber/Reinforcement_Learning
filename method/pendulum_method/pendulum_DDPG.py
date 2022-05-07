# -----------------------------
# File: AC-DDPG Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.29
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../network/pendulum_net/")
from pendulum_ACnet_DPG_continue import ActorNet, CriticNet

import torch
import torch.nn as nn
import numpy as np


# Deep Deterministic Policy Gradient
class DDPG(object):
    def __init__(self, state_dim, action_dim, action_bound, args):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = args.memory_capacity  # 经验回放缓冲区大小
        self.replacement = args.replacement  # 控制网络更新的方法
        self.t_replace_counter = 0  # 更新计数器
        self.gamma = args.gamma
        self.lr_a = args.lr_a
        self.lr_c = args.lr_c
        self.batch_size = args.batch  # 每次从经验回放缓冲区读取的数据大小

        # 记忆库
        self.memory = np.zeros((args.memory_capacity, state_dim * 2 + action_dim + 1))
        self.pointer = 0  # 数据条数记数
        # 定义 Actor 网络
        self.actor = ActorNet(state_dim, action_dim, action_bound)
        self.actor_target = ActorNet(state_dim, action_dim, action_bound)
        # 定义 Critic 网络
        self.critic = CriticNet(state_dim, action_dim)
        self.critic_target = CriticNet(state_dim, action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=args.lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=args.lr_c)
        # 选取损失函数
        self.mse_loss = nn.MSELoss()

    # 从经验回放缓冲区随机采样batch size条数据
    def sample(self):
        indices = np.random.choice(self.memory_capacity, size=self.batch_size)
        return self.memory[indices, :]

    # 通过actor网络输出确定动作
    def choose_action(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):

        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])  # 以tau为权重平滑更新actor网络的weight
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])  # 以tau为权重平滑更新actor网络的bias
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])  # 以tau为权重平滑更新critic网络的weight
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])  # 以tau为权重平滑更新critic网络的bias

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1

        # 从记忆库中采样batch size条data
        bm = self.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)  # critic有两个输入,分别是state和action,分别通过各自的网络处理后,最后一层会将其拼接
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()

        # 训练critic
        a_ = self.actor_target(bs_)
        q_ = self.critic_target(bs_, a_)
        q_target = br + self.gamma * q_  # TD目标,相当于s状态下真实的value function
        q_eval = self.critic(bs, ba)  # s状态下预测的value function
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    # 存储进经验回放缓冲区
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))  # r是单个值,其他的为list
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1