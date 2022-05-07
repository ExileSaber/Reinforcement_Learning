# -----------------------------
# File: DQN Algorithm in CartPole
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.5
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../method/cartpole_method/")
from cartpole_DQN import DQN


import os
import gym
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cartpole_DQN_main(args, seed):
    """
    pendulum游戏的PPO模型的执行主函数
    :param args: 存储了多个参数
    :param seed: 环境的随机数种子
    :return: 无
    """
    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(args.seed)

    n_states = env.observation_space.shape[0]  # states数量
    n_actions = env.action_space.n  # action数量
    env_a_shape = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape

    agent = DQN(n_states, n_actions, env_a_shape, args)

    all_ep_r = []
    start_time = time.time()

    for episode in range(args.n_episodes):
        ep_r = 0
        s = env.reset()

        # for t in range(args.len_episode):
        while True:
            a = agent.choose_action(s)
            s_, r, done, info = env.step(a)

            # modify the reward based on the environment state
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            # store the transitions of states
            agent.store_transition(s, a, r, s_)

            ep_r += r

            if agent.memory_counter > args.memory_capacity:
                agent.learn()

            if done:
                break

            s = s_

        print('Seed {:03d} | Episode {:03d} | Reward:{:.03f}'.format(seed, episode, ep_r))

        if episode == 0:
            all_ep_r.append(ep_r.__float__())
        else:
            all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r.__float__() * 0.1)  # 平滑

    # 存储运行时间
    if args.save_running_time is True:
        running_time = time.time() - start_time
        time_dir = '../../result/cartpole_result/cartpole_running_time.csv'

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
        dir = '../../result/cartpole_result/cartpole_result.csv'

        new_columns = 'DQN_' + str(seed)
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