# RL Group 的强化学习仓库

---

**创建时间: 2022-4-28**

**创建者: 李蔚民**

**维护者: 李蔚民, 冯路飞, 魏小林** 

**使用框架: Pytorch 1.8.1, gym 0.21.0**

<br>

## 文件框架

---

├── data_analyse 对实验结果数据的分析绘图<br>
├── environment 交互环境<br>
├── method 深度强化学习模型<br>
├── model_saved 存储的训练好的模型<br>
├── network 网络结构<br>
├── result 保存的模型输出数据<br>
├── single_method&start 单独的可执行训练的代码<br>
└── start_code 对应于method文件中的模型训练<br>

<br>

## 实现的内容

---

### 包含的环境

1. Flappy Bird


2. Pendulum


3. CartPole


4. Olympics Curling


5. Multi-Agent-Reinforcement-Learning-Environment


6. DeepMind Lab（3D迷宫环境）

<br>

### 各环境实现了的模型

1. Flappy Bird
   1. DQN


2. Pendulum
   1. DDPG
   2. PPO
   3. A3C(暂存在问题)


3. CartPole
   1. DQN


4. Olympics Curling
   1. PPO


5. Multi-Agent-Reinforcement-Learning-Environment
   1. Single Catch Pigs
      1. DQN

<br>

## 待完成的工作

--- 

- [ ] 扩展实现更多环境
- [ ] 扩展实现更多模型
- [ ] 构建自己的模型
