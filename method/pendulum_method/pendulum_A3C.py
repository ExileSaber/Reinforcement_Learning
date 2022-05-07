# -----------------------------
# File: A3C Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.5
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------

import sys
sys.path.append("../../network/pendulum_net/")
from pendulum_ACnet_SPG_continue import ActorNet, CriticNet

import os
import gym
import torch
import time
import torch.nn as nn
import numpy as np
import pandas as pd


def loss_func(args, values, rewards):

    targets = torch.tensor(rewards).to(args.device) + args.gamma * values[1:]
    v = values[:-1]

    # print(targets)
    # print(v)

    loss_func = nn.MSELoss()
    critic_loss = loss_func(v, targets)  # targets是真实的value function

    actor_loss = -torch.mean(values)

    return actor_loss, critic_loss


def worker(shared_actor_model, shared_actor_optimizer, shared_critic_model, shared_critic_optimizer,
           rank, args, info):
    print(rank, 'begin')

    env = gym.make('Pendulum-v1')
    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    n_states = env.observation_space.shape[0]  # states数量
    n_actions = env.action_space.shape[0]  # action数量
    bound = env.action_space.high[0]  # 动作空间的最大值,用于调整网络输出结果的区间范围

    # 初始化模型
    actor_model = ActorNet(n_states, bound).to(args.device)
    critic_model = CriticNet(n_states).to(args.device)

    # 初始状态
    state = torch.tensor(env.reset()).to(args.device)

    # 该线程开始时间
    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done = 0, 0, 0, True

    while info['frames'][0] <= args.max_frame:  # 所有线程的actor最大探索帧数
        actor_model.load_state_dict(shared_actor_model.state_dict())  # 导入主模型参数
        critic_model.load_state_dict(shared_critic_model.state_dict())  # 导入主模型参数

        values, actions, rewards = [], [], []

        for _ in range(args.update_freq):  # 每4个frame更新1次
            episode_length += 1
            mu, sigma = actor_model(state)
            value = critic_model(state)

            dist = torch.distributions.Normal(mu, sigma)  # 生成特定的分布
            action = dist.sample().cpu()  # 根据概率分布采样生成action动作
            action = np.clip(action, -bound, bound)  # 调整动作区间

            state, reward, done, _ = env.step(action)

            state = torch.tensor(state).to(args.device)
            epr += reward
            reward = (reward + 8) / 8
            done = done or episode_length >= args.len_episode  # 每局游戏最大帧长度1e4

            info['frames'].add_(1)  # torch.tensor().add()用完不改变原值，add_()会改变原值
            num_frames = int(info['frames'].item())

            if done:
                info['episodes'] += 1
                # 以下代码在做滑动平均
                interp = 1 if info['episodes'][0] == 1 else 0.1
                info['run_epr'].mul_(1 - interp).add_(interp * epr)
                info['run_loss'].mul_(1 - interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 5:  # 第0个worker每分钟输出1次训练信息
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                print('time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                      .format(elapsed, info['episodes'].item(), num_frames / 1e6,
                              info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(env.reset()).to(args.device)

            values.append(value)
            actions.append(action)
            rewards.append(reward)

        # next_value = torch.zeros(1, 1) if done else critic_model((state.unsqueeze(0)))[0]
        next_value = critic_model((state.unsqueeze(0)))[0]
        values.append(next_value.detach())

        a_loss, c_loss = loss_func(args, torch.cat(values), np.asarray(rewards))
        eploss += a_loss.item() + c_loss.item()

        # print(a_loss)
        # print(c_loss)

        shared_actor_optimizer.zero_grad()
        a_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(actor_model.parameters(), 40)
        for param, shared_param in zip(actor_model.parameters(), shared_actor_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        shared_actor_optimizer.step()  # 在主模型上更新梯度

        shared_critic_optimizer.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic_model.parameters(), 40)
        for param, shared_param in zip(critic_model.parameters(), shared_critic_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad
        shared_critic_optimizer.step()  # 在主模型上更新梯度

    print(info['run_epr'].item())

    if rank == (args.processes - 1):
        # 存储运行时间
        if args.save_running_time is True:
            running_time = time.time() - start_time
            time_dir = '../../result/pendulum_result/pendulum_running_time.csv'

            new_columns = 'A3C'
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
            new_columns = 'A3C_' + str(rank)
            if new_columns in columns:
                del df_old[new_columns]

            data_dict = {new_columns: info['run_epr'].item()}
            df = pd.DataFrame(data_dict)
            df = pd.concat([df_old, df], axis=1)

        else:
            data_dict = {args.method: info['run_epr'].item()}
            df = pd.DataFrame(data_dict)

        df.to_csv(dir, index=None)

    print(rank, 'finish')