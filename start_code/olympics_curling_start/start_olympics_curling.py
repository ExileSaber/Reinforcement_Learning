# -----------------------------
# File: PPO Algorithm in Olympics Curling
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------

import argparse  # 参数接口


from olympics_curling_PPO_discrete_main import PPO_discrete_main


parser = argparse.ArgumentParser()
parser.add_argument('--game_name', default="olympics-curling", type=str)
parser.add_argument('--algo', default="ppo", type=str, help="ppo")
parser.add_argument('--max_episodes', default=3000, type=int)

parser.add_argument('--seed', default=1, type=int)

parser.add_argument("--save_interval", default=500, type=int)
parser.add_argument("--model_episode", default=0, type=int)

parser.add_argument("--load_model", default=False, type=bool)  # 加是true；不加为false
parser.add_argument("--load_run", default=2, type=int)
parser.add_argument("--load_episode", default=1300, type=int)

parser.add_argument("--opponent_load_run", default=3, type=int)
parser.add_argument("--opponent_load_episode", default=1300, type=int)

parser.add_argument("--device", default='cpu', type=str)

parser.add_argument("--clip_param", default=0.2, type=float)
parser.add_argument("--max_grad_norm", default=0.5, type=float)
parser.add_argument("--ppo_update_time", default=10, type=int)
parser.add_argument("--buffer_capacity", default=1000, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--lr", default=0.0001, type=float)

parser.add_argument("--action_space", default=36, type=int)
parser.add_argument("--state_space", default=900, type=int)

args = parser.parse_args()


if __name__ == '__main__':
    PPO_discrete_main(args)
