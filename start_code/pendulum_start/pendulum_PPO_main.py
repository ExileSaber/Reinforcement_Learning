# -----------------------------
# File: AC-PPO Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.29
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../method/pendulum_method/")
from pendulum_PPO import PPO


import os
import gym
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pendulum_PPO_main(args, seed):
    """
    pendulum游戏的PPO模型的执行主函数
    :param args: 存储了多个参数
    :param seed: 环境的随机数种子
    :return: 无
    """
    env = gym.make('Pendulum-v1')
    env.seed(seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]  # states数量
    n_actions = env.action_space.shape[0]  # action数量
    bound = env.action_space.high[0]  # 动作空间的最大值,用于调整网络输出结果的区间范围

    agent = PPO(n_states, n_actions, bound, args)

    all_ep_r = []
    start_time = time.time()

    for episode in range(args.n_episodes):  # 执行轮数
        ep_r = 0  # 存储总的reward
        s = env.reset()
        states, actions, rewards = [], [], []
        for t in range(args.len_episode):  # 每一轮最大步长
            a = agent.choose_action(s)
            s_, r, done, _ = env.step(a)
            ep_r += r

            a = a[0].detach().cpu().numpy()

            states.append(s)
            actions.append(a)
            rewards.append((r + 8) / 8)  # 参考了网上的做法

            s = s_

            if (t + 1) % args.batch == 0 or t == args.len_episode - 1:  # N步更新
                states = np.array(states)
                actions = np.array(actions)
                rewards = np.array(rewards)

                targets = agent.discount_reward(rewards, s_)  # 奖励回溯,直接用args.batch个step的完整路径进行更新
                agent.update(states, actions, targets)  # 进行actor和critic网络的更新
                states, actions, rewards = [], [], []  # 更新之后直接清空,没有采用经验回放缓冲区的方式

        print('Seed {:03d} | Episode {:03d} | Reward:{:.03f}'.format(seed, episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r.__float__())
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r.__float__() * 0.1)  # 平滑

    # 存储运行时间
    if args.save_running_time is True:
        running_time = time.time() - start_time
        time_dir = '../../result/pendulum_result/pendulum_running_time.csv'

        new_columns = 'PPO'
        if os.path.exists(time_dir):
            df_time = pd.read_csv(time_dir)
            columns = df_time.columns
            if new_columns in columns:
                index = df_time[new_columns].last_valid_index() + 1
                df_time.loc[index, new_columns] = running_time
            else:
                df_time.loc[0, new_columns] = running_time
        else:
            data_dict = {new_columns: running_time}
            df_time = pd.DataFrame([data_dict])

        df_time.to_csv(time_dir, index=None)

    # 存储平滑损失
    if args.save_reward_list is True:
        dir = '../../result/pendulum_result/pendulum_result.csv'
        if os.path.exists(dir):
            df_old = pd.read_csv(dir)
            columns = df_old.columns
            new_columns = 'PPO_' + str(seed)
            if new_columns in columns:
                del df_old[new_columns]

            data_dict = {new_columns: all_ep_r}
            df = pd.DataFrame(data_dict)
            df = pd.concat([df_old, df], axis=1)

        else:
            data_dict = {args.method: all_ep_r}
            df = pd.DataFrame(data_dict)

        df.to_csv(dir, index=None)

    # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.show()