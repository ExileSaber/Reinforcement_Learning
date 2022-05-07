# -----------------------------
# File: Deep Q-Learning Algorithm in Flappy Bird
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.28
# Pytorch Version: 1.8.1
# -----------------------------

import sys
import cv2
import numpy as np

sys.path.append("../../method/flappy_bird_method")
from flappy_bird_DQN import BrainDQNMain, preprocess

sys.path.append("../../environment/flappy_bird_env/game")
import wrapped_flappy_bird as game


ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1000.  # 前OBSERVE轮次，不对网络进行训练，只是收集数据，存到记忆库中
# 第OBSERVE到OBSERVE+EXPLORE轮次中，对网络进行训练，且对epsilon进行退火，逐渐减小epsilon至FINAL_EPSILON
# 当到达EXPLORE轮次时，epsilon达到最终值FINAL_EPSILON，不再对其进行更新
EXPLORE = 2000000.
FINAL_EPSILON = 0.  # epsilon的最终值
INITIAL_EPSILON = 0.  # epsilon的初始值
REPLAY_MEMORY = 50000  # 记忆库
BATCH_SIZE = 32  # 训练批次
FRAME_PER_ACTION = 1  # 每隔FRAME_PER_ACTION轮次，就会有epsilon的概率进行探索
UPDATE_TIME = 100  # 每隔UPDATE_TIME轮次，对target网络的参数进行更新
DEVICE = 'cuda'

EPOCH = 1000000


if __name__ == '__main__':
    brain = BrainDQNMain(ACTIONS, GAMMA, OBSERVE, EXPLORE, FINAL_EPSILON, INITIAL_EPSILON, REPLAY_MEMORY, BATCH_SIZE,
                         FRAME_PER_ACTION, UPDATE_TIME, DEVICE)
    flappyBird = game.GameState()
    action0 = np.array([1, 0])  # 一个随机动作
    # 执行一个动作，获得执行动作后的下一帧图像、reward、游戏是否终止的标志
    observation0, reward0, terminal = flappyBird.frame_step(action0)
    # 将彩色图像转化为灰度值图像
    observation0 = cv2.cvtColor(cv2.resize(observation0, (80, 80)), cv2.COLOR_BGR2GRAY)
    # 将灰度图像转化为二值图像
    ret, observation0 = cv2.threshold(observation0, 1, 255, cv2.THRESH_BINARY)
    # 将一帧图片重复4次，每一张图片为一个通道，变成通道为4的输入，即初始输入是4帧相同的图片
    brain.setInitState(observation0)
    print(brain.currentState.shape)

    max_step = 0
    step = 0
    epoch = 0

    while True:
        # 获取下一个动作
        action = brain.getAction()
        # 执行动作，获得执行动作后的下一帧图像、reward、游戏是否终止的标志
        nextObservation, reward, terminal = flappyBird.frame_step(action)

        # 将一帧彩色图像处理成黑白的二值图像
        nextObservation = preprocess(nextObservation)
        # print(nextObservation.shape)
        brain.setPerception(nextObservation, action, reward, terminal)

        step += 1
        if terminal:
            epoch += 1
            if step > max_step:
                max_step = step
            print('epoch {} play step: '.format(epoch), step, ' max step: ', max_step)
            print('-' * 20)
            step = 0

            if epoch >= EPOCH:
                break


