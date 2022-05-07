# -----------------------------
# File: AC-PPO Algorithm in Pendulum
# Author: Others
# Modify: Li Weimin
# Date: 2022.4.29
# Pytorch Version: 1.8.1
# gym Version : 0.21.0
# -----------------------------


import argparse
from pendulum_DDPG_main import pendulum_DDPG_main
from pendulum_PPO_main import pendulum_PPO_main
from pendulum_Simple_main import pendulum_Simple_main
from pendulum_A3C_main import pendulum_A3C_main


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, default=8)

    parser.add_argument('--n_episodes', type=int, default=1000)
    parser.add_argument('--len_episode', type=int, default=200)
    parser.add_argument('--max_frame', type=int, default=2000000)
    parser.add_argument('--update_freq', default=4, type=int, help='every 4 frames update once')

    parser.add_argument('--lr_a', type=float, default=0.0001)
    parser.add_argument('--lr_c', type=float, default=0.0001)

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)

    parser.add_argument('--c_update_steps', type=int, default=10)
    parser.add_argument('--a_update_steps', type=int, default=10)

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--memory_capacity', type=int, default=10000)
    parser.add_argument('--replacement', type=list, default=[dict(name='soft', tau=0.01), dict(name='hard', rep_iter=600)][0])  # you can try different target replacement strategies)

    parser.add_argument('--save_reward_list', type=bool, default=False)
    parser.add_argument('--save_running_time', type=bool, default=False)
    # parser.add_argument('--method', type=str, default='PPO')
    args = parser.parse_args()

    start_seed = 0
    end_seed = 1

    # # PPO训练
    for seed in range(start_seed, end_seed):
        pendulum_PPO_main(args, seed)

    # Simple训练
    # 当同样的结构不使用PPO时,容易出现梯度爆炸的情况,网络输出结果为 Nan
    for seed in range(start_seed, end_seed):
        pendulum_Simple_main(args, seed)

    # DDPG训练
    for seed in range(start_seed, end_seed):
        pendulum_DDPG_main(args, seed)

    # =====================================
    # A3C训练,还未调试好,不能收敛,并且cpu温度较高
    # =====================================
    # pendulum_A3C_main(args)
