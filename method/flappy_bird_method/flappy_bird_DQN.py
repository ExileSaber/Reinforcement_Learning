# -----------------------------
# File: Deep Q-Learning Algorithm in Flappy Bird
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.28
# Pytorch Version: 1.8.1
# -----------------------------

import cv2
import sys
import os

sys.path.append("../../network/flappy_bird_net")
from flappy_bird_net import DeepNetWork

import random
import numpy as np
from collections import deque
import torch
from torch.autograd import Variable
import torch.nn as nn


# 将一帧彩色图像处理成黑白的二值图像
def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (1, 80, 80))


class BrainDQNMain(object):
    def save(self):
        print("save model param")
        torch.save(self.Q_net.state_dict(), '../../model_saved/flappy_bird_saved/params3.pth')

    def load(self):
        if os.path.exists("../../model_saved/flappy_bird_saved/params3.pth"):
            print("load model param")
            self.Q_net.load_state_dict(torch.load('../../model_saved/flappy_bird_saved/params3.pth'))
            self.Q_netT.load_state_dict(torch.load('../../model_saved/flappy_bird_saved/params3.pth'))

    def __init__(self, actions, gamma, observe, explore, final_epsilon, initial_epsilon, replay_memory, batch_size,
                 frame_pre_action, update_time, device):
        """

        :param actions: number of valid actions
        :param gamma: decay rate of past observations
        :param observe: 前 observe 轮次，不对网络进行训练，只是收集数据，存到记忆库中
        :param explore: 第 observe 到 observe + explore 轮次中，对网络进行训练，且对 epsilon 进行退火，逐渐减小 epsilon 至 final_epsilon, 当到达 explore 轮次时，epsilon 达到最终值 final_epsilon，不再对其进行更新
        :param final_epsilon: epsilon 的最终值
        :param initial_epsilon: epsilon 的初始值
        :param replay_memory: 记忆库大小
        :param batch_size: 每次训练提取的 batch 大小
        :param frame_pre_action: 每隔 frame_pre_action 轮次，就会有 epsilon 的概率进行探索
        :param update_time: 每隔 update_time 轮次，对target网络的参数进行更新
        :param device: 训练环境,是否使用显卡
        
        """
        # 在每个timestep下agent与环境交互得到的转移样本 (st,at,rt,st+1) 储存到回放记忆库，
        # 要训练时就随机拿出一些（minibatch）数据来训练，打乱其中的相关性
        self.replayMemory = deque()  # init some parameters
        self.timeStep = 0
        # 有epsilon的概率，随机选择一个动作，1-epsilon的概率通过网络输出的Q（max）值选择动作
        self.epsilon = initial_epsilon
        # 初始化部分参数
        self.actions = actions
        self.gamma = gamma
        self.observe = observe
        self.explore = explore
        self.final_epsilon = final_epsilon
        self.replay_memory = replay_memory
        self.batch_size = batch_size
        self.frame_pre_action = frame_pre_action
        self.update_time = update_time
        self.device = device

        # 当前值网络
        self.Q_net = DeepNetWork().to(self.device)
        # 目标值网络
        self.Q_netT = DeepNetWork().to(self.device)
        # 加载训练好的模型，在训练的模型基础上继续训练
        self.load()
        # 使用均方误差作为损失函数
        self.loss_func = nn.MSELoss()
        LR = 1e-6
        self.optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=LR)

    # 使用minibatch训练网络
    def train(self):  # Step 1: obtain random minibatch from replay memory
        # 从记忆库中随机获得BATCH_SIZE个数据进行训练
        minibatch = random.sample(self.replayMemory, self.batch_size)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]  # Step 2: calculate y
        # y_batch用来存储reward
        y_batch = np.zeros([self.batch_size, 1])
        nextState_batch = np.array(nextState_batch)  # print("train next state shape")
        # print(nextState_batch.shape)
        nextState_batch = torch.Tensor(nextState_batch).to(self.device)
        action_batch = np.array(action_batch)
        # 每个action包含两个元素的数组，数组必定是一个1，一个0，最大值的下标也就是该action的下标
        index = action_batch.argmax(axis=1)

        # print("Train minibatch action " + str(index))

        index = np.reshape(index, [self.batch_size, 1])
        # 预测的动作的下标
        action_batch_tensor = torch.LongTensor(index).to(self.device)
        # 使用target网络，预测nextState_batch的动作
        QValue_batch = self.Q_netT(nextState_batch)
        QValue_batch = QValue_batch.detach().cpu().numpy()
        # 计算每个state的reward
        for i in range(0, self.batch_size):
            # terminal是结束标志
            terminal = minibatch[i][4]
            if terminal:
                y_batch[i][0] = reward_batch[i]
            else:
                # 这里的QValue_batch[i]为数组，大小为所有动作集合大小，QValue_batch[i],代表
                # 做所有动作的Q值数组，y计算为如果游戏停止，y=rewaerd[i],如果没停止，则y=reward[i]+gamma*np.max(Qvalue[i])
                # 代表当前y值为当前reward+未来预期最大值*gamma(gamma:经验系数)
                #网络的输出层的维度为2，将输出值中的最大值作为Q值
                y_batch[i][0] = reward_batch[i] + self.gamma * np.max(QValue_batch[i])

        y_batch = np.array(y_batch)
        y_batch = np.reshape(y_batch, [self.batch_size, 1])
        state_batch_tensor = Variable(torch.Tensor(state_batch)).to(self.device)
        y_batch_tensor = Variable(torch.Tensor(y_batch)).to(self.device)
        y_predict = self.Q_net(state_batch_tensor).gather(1, action_batch_tensor)
        loss = self.loss_func(y_predict, y_batch_tensor)

        # print("Loss is " + str(loss))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 每隔UPDATE_TIME轮次，用训练的网络的参数来更新target网络的参数
        if self.timeStep % self.update_time == 0:
            self.Q_netT.load_state_dict(self.Q_net.state_dict())
            self.save()

    # 更新记忆库，若轮次达到一定要求则对网络进行训练
    def setPerception(self, nextObservation, action, reward, terminal):  # print(nextObservation.shape)
        # 每个state由4帧图像组成
        # nextObservation是新的一帧图像,记做5。currentState包含4帧图像[1,2,3,4]，则newState将变成[2,3,4,5]
        newState = np.append(self.currentState[1:, :, :], nextObservation,
                             axis=0)  # newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
        # 将当前状态存入记忆库
        self.replayMemory.append((self.currentState, action, reward, newState, terminal))
        # 若记忆库已满，替换出最早进入记忆库的数据
        if len(self.replayMemory) > self.replay_memory:
            self.replayMemory.popleft()
        # 在训练之前，需要先观察OBSERVE轮次的数据，经过收集OBSERVE轮次的数据之后，开始训练网络
        if self.timeStep > self.observe:  # Train the network
            self.train()

        # print info
        state = ""
        # 在前OBSERVE轮中，不对网络进行训练，相当于对记忆库replayMemory进行填充数据
        if self.timeStep <= self.observe:
            state = "observe"
        elif self.timeStep > self.observe and self.timeStep <= self.observe + self.explore:
            state = "explore"
        else:
            state = "train"

        # print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

        self.currentState = newState
        self.timeStep += 1

    # 获得下一步要执行的动作
    def getAction(self):
        currentState = torch.Tensor([self.currentState]).to(self.device)
        # QValue为网络预测的动作
        QValue = self.Q_net(currentState)[0]
        action = np.zeros(self.actions)
        # self.frame_pre_action=1表示每一步都有可能进行探索
        if self.timeStep % self.frame_pre_action == 0:
            if random.random() <= self.epsilon:  # 有epsilon得概率随机选取一个动作
                action_index = random.randrange(self.actions)

                # print("Choose RANDOM action " + str(action_index))

                action[action_index] = 1
            else:  # 1-epsilon的概率通过神经网络选取下一个动作
                action_index = np.argmax(QValue.detach().cpu().numpy())

                # print("choose DQN value action " + str(action_index))

                action[action_index] = 1
        else:  # 程序貌似不会走到这里
            action[0] = 1  # do nothing

        # 随着迭代次数增加，逐渐减小episilon
        if self.epsilon > self.final_epsilon and self.timeStep > self.observe:
            self.epsilon -= (self.epsilon - self.final_epsilon) / self.explore
        return action

    # 初始化状态
    def setInitState(self, observation):
        # 增加一个维度，observation的维度是80x80，讲过stack()操作之后，变成4x80x80
        self.currentState = np.stack((observation, observation, observation, observation), axis=0)
        print(self.currentState.shape)

