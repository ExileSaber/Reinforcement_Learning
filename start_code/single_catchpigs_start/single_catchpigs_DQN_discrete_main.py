# -----------------------------
# File: DQN Algorithm in Single Catch Pigs
# Author: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------

import sys
sys.path.append('../../environment/Multi-Agent-Reinforcement-Learning-Environment/env_SingleCatchPigs')
sys.path.append('../../method/single_catchpigs_method')

from env_SingleCatchPigs import EnvSingleCatchPigs
from single_catchpigs_DQN_discrete import DQN

import os
import gym
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def single_catchpigs_DQN_discrete_main(args):
    """
    pendulum游戏的PPO模型的执行主函数
    :param args: 存储了多个参数
    :param seed: 环境的随机数种子
    :return: 无
    """
    env = EnvSingleCatchPigs(7)
    env.set_agent_at([2, 2], 0)
    env.set_pig_at([4, 4], 0)

    torch.manual_seed(args.seed)

    n_states = 2 + 2 + 1  # states数量
    n_actions = 5 # action数量

    agent = DQN(n_states, n_actions, args)

    all_ep_r = []
    start_time = time.time()

    for episode in range(args.n_episodes):
        ep_r = 0
        env.reset()

        s = env.agt1_pos + env.pig_pos + [env.agt1_ori]

        for t in range(args.len_episode):
        # while True:
            if args.render is True:
                env.render()
            a = agent.choose_action(s)
            r, done = env.step(a)
            s_ = env.agt1_pos + env.pig_pos + [env.agt1_ori]

            # store the transitions of states
            agent.store_transition(s, a, r, s_)

            ep_r += r

            if agent.memory_counter > args.memory_capacity:
                agent.learn()

            if done:
                break

            s = s_

        print('Episode {:03d} | Reward:{:.03f}'.format(episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r.__float__())
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r.__float__() * 0.1)  # 平滑

    # 存储运行时间
    if args.save_running_time is True:
        running_time = time.time() - start_time
        time_dir = '../../result/single_catchpigs_result/single_catchpigs_running_time.csv'

        new_columns = 'DQN'
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
        dir = '../../result/single_catchpigs_result/single_catchpigs_result.csv'

        new_columns = 'DQN'
        if os.path.exists(dir):
            df_old = pd.read_csv(dir)
            columns = df_old.columns

            if new_columns in columns:
                del df_old[new_columns]

            data_dict = {new_columns: all_ep_r}
            df = pd.DataFrame(data_dict)
            df = pd.concat([df_old, df], axis=1)

        else:
            data_dict = {new_columns: all_ep_r}
            df = pd.DataFrame(data_dict)

        df.to_csv(dir, index=None)

    # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.show()