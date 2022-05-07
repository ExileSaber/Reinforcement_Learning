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
from pendulum_DDPG import DDPG

import os
import gym
import torch
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


def pendulum_DDPG_main(args, seed):
    env = gym.make('Pendulum-v1')
    env.seed(seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]
    # print(n_states)
    n_actions = env.action_space.shape[0]
    # print(n_actions)
    bound = env.action_space.high[0]
    low_bound = env.action_space.low

    agent = DDPG(n_states, n_actions, bound, args)
    var = 3

    all_ep_r = []
    start_time = time.time()

    for episode in range(args.n_episodes):
        ep_r = 0
        s = env.reset()

        for t in range(args.len_episode):
            a = agent.choose_action(s)
            a = np.clip(np.random.normal(a, var), low_bound, bound)  # 在动作选择上添加随机噪声

            s_, r, done, _ = env.step(a)
            ep_r += r

            # a = a[0].detach().cpu().numpy()

            agent.store_transition(s, a, (r + 8) / 8, s_)  # store the transition to memory

            if agent.pointer > args.memory_capacity:
                var *= 0.9995  # decay the exploration controller factor
                agent.learn()

            s = s_

        print('Seed {:03d} | Episode {:03d} | Reward:{:.03f}'.format(seed, episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r.__float__())
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r.__float__() * 0.1)  # 平滑

    # 存储运行时间
    if args.save_running_time is True:
        running_time = time.time() - start_time
        time_dir = '../../result/pendulum_result/pendulum_running_time.csv'

        new_columns = 'DDPG'
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

    if args.save_reward_list is True:
        dir = '../../result/pendulum_result/pendulum_result.csv'
        if os.path.exists(dir):
            df_old = pd.read_csv(dir)
            columns = df_old.columns
            new_columns = 'DDPG_' + str(seed)
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