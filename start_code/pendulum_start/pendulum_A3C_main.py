# -----------------------------
# File: A3C-PPO Algorithm in Pendulum
# Author: Li Weimin
# Date: 2022.5.5
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../method/pendulum_method/")
from pendulum_A3C import worker

sys.path.append("../../network/pendulum_net/")
from pendulum_ACnet_SPG_continue import ActorNet, CriticNet

import os
import gym
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp


class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()


def pendulum_A3C_main(args):
    if sys.version_info[0] > 2: # 判断python版本 3.X
        mp.set_start_method('spawn', force=True) # 多进程训练

    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux! Or else you get a deadlock in conv2d"

    env = gym.make('Pendulum-v1')
    n_states = env.observation_space.shape[0]  # states数量
    n_actions = env.action_space.shape[0]  # action数量
    bound = env.action_space.high[0]  # 动作空间的最大值,用于调整网络输出结果的区间范围

    torch.manual_seed(args.seed)
    shared_actor_model = ActorNet(n_states, bound).to(args.device).share_memory()
    shared_actor_optimizer = SharedAdam(shared_actor_model.parameters(), lr=args.lr_a)

    shared_critic_model = CriticNet(n_states).to(args.device).share_memory()
    shared_critic_optimizer = SharedAdam(shared_critic_model.parameters(), lr=args.lr_c)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

    processes = []
    for rank in range(args.processes):
        p = mp.Process(target=worker, args=(shared_actor_model, shared_actor_optimizer,
                                            shared_critic_model, shared_critic_optimizer,
                                            rank, args, info))
        p.start()
        processes.append(p)
    for p in processes: p.join()
