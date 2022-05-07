# -----------------------------
# File: DQN Algorithm in Single Catch Pigs
# Author: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append('../../network/single_catchpigs_net')
from single_catchpigs_DQNnet_discrete import DeepNetWork


# 多智能体模型,暂时不深入研究
class DQN(object):
    def __init__(self, n_states, n_actions, args):
        self.args = args
        self.n_states = n_states
        self.n_actions = n_actions
        # -----------Define 2 networks (target and training)------#
        self.eval_net = DeepNetWork(n_states, n_actions).to(args.device)
        self.target_net = DeepNetWork(n_states, n_actions).to(args.device)
        # Define counter, memory size and loss function
        self.learn_step_counter = 0  # count the steps of learning process
        self.memory_counter = 0  # counter used for experience replay buffer

        # ----Define the memory (or the buffer), allocate some space to it. The number
        # of columns depends on 4 elements, s, a, r, s_, the total is N_STATES*2 + 2---#
        self.memory = np.zeros((args.memory_capacity, n_states * 2 + 2))  # 这里第二维分配了10个维度

        # ------- Define the optimizer------#
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=args.lr)

        # ------Define the loss function-----#
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        # This function is used to make decision based upon epsilon greedy

        x = torch.unsqueeze(torch.FloatTensor(x).to(self.args.device), 0)  # add 1 dimension to input state x，对第一个参数插入一个维度，0表示一行插入一个维度，1表示一列插入一个维度
        # input only one sample
        if np.random.uniform() < self.args.choice_action_epsilon:  # greedy，以EPSILON的概率前向传播
            # use epsilon-greedy approach to take action
            actions_value = self.eval_net.forward(x)  # 训练网络前向传播
            # print(torch.max(actions_value, 1))
            # torch.max() returns a tensor composed of max value along the axis=dim and corresponding index
            # what we need is the index in this function, representing the action of cart.
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 返回最大值对应的索引
        else:  # random
            action = np.random.randint(0, self.n_actions)  # 随机产生一个action
        return action

    def store_transition(self, s, a, r, s_):
        # This function acts as experience replay buffer
        transition = np.hstack((s, [a, r], s_))  # horizontally stack these vectors，水平叠加这些向量
        # if the capacity is full, then use index to replace the old memory with new one
        index = self.memory_counter % self.args.memory_capacity  # 经验回放计数器，判断是否已经存满了一轮
        self.memory[index, :] = transition  # 存入经验回放池
        self.memory_counter += 1

    def learn(self):
        # Define how the whole DQN works including sampling batch of experiences,
        # when and how to update parameters of target network, and how to implement
        # backward propagation.

        # update the target network every fixed steps
        if self.learn_step_counter % self.args.update_steps == 0:
            # Assign the parameters of eval_net to target_net
            # torch.nn.Module模块中的state_dict变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量，
            # 需要注意的是torch.nn.Module模块中的state_dict只包含卷积层和全连接层的参数，当网络中存在batchnorm时，
            # 例如vgg网络结构，torch.nn.Module模块中的state_dict也会存放batchnorm's running_mean
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(self.args.memory_capacity, self.args.batch)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.n_states]))
        # convert long int type to tensor
        b_a = Variable(torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.n_states:]))

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # print(q_eval)
        # calculate the q value of next state
        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        # print(q_next)
        # q_next.max(1) returns the max value along the axis=1 and its corresponding index
        q_target = b_r + self.args.gamma * q_next.max(1)[0].view(self.args.batch, 1)  # (batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step
