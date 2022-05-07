# -----------------------------
# File: PPO Algorithm in Olympics Curling
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------

import datetime
import math

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import os

import sys
sys.path.append('../../environment/olympics_curling_env/env')
sys.path.append('../../method/olympics_curling_method')

# 导入更多的数据类型：deque 双端队列，支持从两端添加和删除元素；namedtuple 标准的元组使用数值索引来访问其成员，也可以通过定义的名字
from collections import deque, namedtuple


from chooseenv import make
from log_path import make_logpath, save_config
from olympics_curling_PPO_discrete import PPO
from olympics_curling_random import random_agent


RENDER = True
# 动作行为字典，用于选择动作
action_choose = 16
# 动作行为字典，用于选择动作
actions_map = {0: [-100, -30], 1: [-100, -18], 2: [-100, -6], 3: [-100, 6], 4: [-100, 18], 5: [-100, 30],
               6: [-40, -30], 7: [-40, -18], 8: [-40, -6], 9: [-40, 6], 10: [-40, 18], 11: [-40, 30],
               12: [20, -30], 13: [20, -18], 14: [20, -6], 15: [20, 6], 16: [20, 18], 17: [20, 30],
               18: [80, -30], 19: [80, -18], 20: [80, -6], 21: [80, 6], 22: [80, 18], 23: [80, 30],
               24: [140, -30], 25: [140, -18], 26: [140, -6], 27: [140, 6], 28: [140, 18], 29: [140, 30],
               30: [200, -30], 31: [200, -18], 32: [200, -6], 33: [200, 6], 34: [200, 18], 35: [200, 30]}          #dicretise action space  #dicretise action space


# 计算两点距离
def compute_distance(p1, p2):
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx**2 + dy**2)


# 训练主函数
def PPO_discrete_main(args):
    print("==algo: ", args.algo)
    print(f'device: {args.device}')
    print(f'model episode: {args.model_episode}')
    print(f'save interval: {args.save_interval}')

    env = make(args.game_name)

    num_agents = env.n_player
    print(f'Total agent number: {num_agents}')

    ctrl_agent_index = 1
    print(f'Agent control by the actor: {ctrl_agent_index}')

    ctrl_agent_num = 1

    width = env.env_core.view_setting['width']+2*env.env_core.view_setting['edge']
    height = env.env_core.view_setting['height']+2*env.env_core.view_setting['edge']
    print(f'Game board width: {width}')
    print(f'Game board height: {height}')

    act_dim = env.action_dim
    obs_dim = 30*30
    print(f'action dimension: {act_dim}')
    print(f'observation dimension: {obs_dim}')

    torch.manual_seed(args.seed)  # 控制随机种子
    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)

    writer = SummaryWriter(os.path.join(str(log_dir), "{}_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.algo)))  # 用于存储标量，之后可以用浏览器访问查看
    save_config(args, log_dir)  # 存储参数文件

    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    if args.load_model:
        model = PPO(args)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir,episode=args.load_episode)
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])
    else:
        model = PPO(args, run_dir)
        # model = PPO_new(run_dir)
        # 定义了一个namedtuple类型，该类型包括 'state', 'action', 'a_log_prob', 'reward', 'next_state', 'done' 6个属性
        Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state', 'done'])

    opponent_agent = random_agent()     #we use random opponent agent here（使用的随机对手智能体）

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        ctrl_agent_index = 0 if ctrl_agent_index == 1 else 1  # 攻守轮换
        if ctrl_agent_index == 0:
            print("我方先手, 紫色为我方")
        else:
            print("我方后手, 绿色为我方")

        # 初始化，这里的两个25*25的矩阵表示的是什么？
        state = env.reset()    #[{'obs':[25,25], "control_player_index": 0}, {'obs':[25,25], "control_player_index": 1}]
        if RENDER:
            env.env_core.render()
        obs_ctrl_agent = np.array(state[ctrl_agent_index]['obs']).flatten()     #[25*25]
        obs_oppo_agent = state[1-ctrl_agent_index]['obs']   #[25,25] 采用随机动作时使用

        episode += 1
        step = 0
        Gt = 0

        while True:
            # 采用随机动作
            # action_opponent = opponent_agent.act(obs_oppo_agent)
            # action_opponent = actions_map[action_choose]
            # action_opponent = [[action_opponent[0]],[action_opponent[1]]]  #here we assume the opponent is not moving in the demo（这里我们假设对手在演示中没有移动）
            # action_opponent = [[90], [0]]
            action_opponent = [[0], [0]]

            action_ctrl_raw, action_prob= model.select_action(obs_ctrl_agent, True)
                            #inference
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]        #wrapping up the action

            action = [action_opponent, action_ctrl] if ctrl_agent_index == 1 else [action_ctrl, action_opponent]

            next_state, reward, done, _, info = env.step(action)

            next_obs_ctrl_agent = next_state[ctrl_agent_index]['obs']
            next_obs_oppo_agent = next_state[1-ctrl_agent_index]['obs']

            step += 1

            if not done:
                post_reward = [-1., -1.]
            else:
                if reward[0] != reward[1]:
                    post_reward = [reward[0]-100, reward[1]] if reward[0]<reward[1] else [reward[0], reward[1]-100]
                else:
                    post_reward=[-1., -1.]

            if env.env_core.current_team == ctrl_agent_index:
                post_reward[ctrl_agent_index] = -compute_distance([300, 500], env.env_core.agent_pos[-1])
                trans = Transition(obs_ctrl_agent, action_ctrl_raw, action_prob, post_reward[ctrl_agent_index],
                                   next_obs_ctrl_agent, done)
                model.store_transition(trans)

            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
            if RENDER:
                env.env_core.render()
            Gt += reward[ctrl_agent_index] if done else -1

            if done:
                win_is = 1 if reward[ctrl_agent_index]>reward[1-ctrl_agent_index] else 0
                win_is_op = 1 if reward[ctrl_agent_index]<reward[1-ctrl_agent_index] else 0
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print("Episode: ", episode, "controlled agent: ", ctrl_agent_index, "; Episode Return: ", Gt,
                      "; win rate(controlled & opponent): ", '%.2f' % (sum(record_win)/len(record_win)),
                      '%.2f' % (sum(record_win_op)/len(record_win_op)), '; Trained episode:', train_count)


                if args.algo == 'ppo' and len(model.buffer) >= args.batch_size:
                    if win_is == 1:
                        model.update(episode)
                        train_count += 1
                    else:
                        model.clear_buffer()

                writer.add_scalar('training Gt', Gt, episode)

                break
        if episode % args.save_interval == 0:
            model.save(run_dir, episode)

