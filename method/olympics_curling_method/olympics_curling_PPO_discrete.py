# -----------------------------
# File: PPO Algorithm in Olympics Curling
# Author: Others
# Modify: Li Weimin
# Date: 2022.5.6
# Pytorch Version: 1.8.1
# -----------------------------

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import datetime
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('../../network/olympics_curling_net')
from olympics_curling_ACnet_discrete import Actor, Critic


class PPO:
    def __init__(self, args, run_dir=None):
        super(PPO, self).__init__()
        self.args = args
        self.actor_net = Actor(args.state_space, args.action_space).to(self.args.device)
        self.critic_net = Critic(args.state_space).to(self.args.device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.args.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.args.lr)

        if run_dir is not None:
            self.writer = SummaryWriter(os.path.join(run_dir, "PPO training loss at {}".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))
        self.IO = True if (run_dir is not None) else False

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.args.device)
        with torch.no_grad():
            action_prob = self.actor_net(state).to(self.args.device)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
            # action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor(np.array([t.state for t in self.buffer]), dtype=torch.float).to(self.args.device)
        action = torch.tensor(np.array([t.action for t in self.buffer]), dtype=torch.long).view(-1, 1).to(self.args.device)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.args.device)

        # print(state)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.args.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(self.args.device)
        # print("The agent is updateing....")
        for i in range(self.args.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.args.batch_size, False):
                # if self.training_step % 1000 == 0:
                #     print('I_ep {} ，train {} times'.format(i_ep, self.training_step))
                # with torch.no_grad():
                # print(index)
                # print(state[index])
                # print("-" * 20)
                # print(state[index].squeeze(1))
                # print("="*20)

                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index].squeeze(1))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index].squeeze(1)).gather(1, action[index])  # new policy

                # print(action[index])
                # print("action_prob: ", action_prob)
                # print('-' * 20)

                ratio = (action_prob / old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.args.clip_param, 1 + self.args.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                # self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.args.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                # self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.args.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

                if self.IO:
                    self.writer.add_scalar('loss/policy loss', action_loss.item(), self.training_step)
                    self.writer.add_scalar('loss/critic loss', value_loss.item(), self.training_step)

        # del self.buffer[:]  # clear experience
        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, 'trained_model')
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        print("run_dir: ", run_dir)
        # base_path = os.path.dirname(os.path.dirname(__file__))
        # print("base_path: ", base_path)
        # algo_path = os.path.join(base_path, 'models/ppo')
        # run_path = os.path.join(algo_path, run_dir)
        # run_path = os.path.join(run_path, 'trained_model')
        run_path = os.path.join(run_dir, 'trained_model')
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.args.device)
            critic = torch.load(model_critic_path, map_location=self.args.device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

