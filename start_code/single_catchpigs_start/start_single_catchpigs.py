# -----------------------------
# File: DQN Algorithm in Single Catch Pigs
# Author: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------


import argparse
from single_catchpigs_DQN_discrete_main import single_catchpigs_DQN_discrete_main



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_episodes', type=int, default=200)
    parser.add_argument('--len_episode', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr_a', type=float, default=0.0001)
    parser.add_argument('--lr_c', type=float, default=0.0001)

    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=10)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--choice_action_epsilon', type=float, default=0.9)

    parser.add_argument('--update_steps', type=int, default=100)
    parser.add_argument('--c_update_steps', type=int, default=10)
    parser.add_argument('--a_update_steps', type=int, default=10)

    parser.add_argument('--device', type=str, default='cpu')

    parser.add_argument('--memory_capacity', type=int, default=2000)
    parser.add_argument('--replacement', type=list, default=[dict(name='soft', tau=0.01), dict(name='hard', rep_iter=600)][0])  # you can try different target replacement strategies)

    parser.add_argument('--save_reward_list', type=bool, default=True)
    parser.add_argument('--save_running_time', type=bool, default=True)
    # parser.add_argument('--method', type=str, default='PPO')

    parser.add_argument('--render', type=bool, default=True)

    args = parser.parse_args()

    single_catchpigs_DQN_discrete_main(args)
