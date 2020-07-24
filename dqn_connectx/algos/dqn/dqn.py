import os
import random
import math
import numpy as np
import gym
import torch
import torch.nn as nn
from torch.optim import Adam
from gym.spaces import Discrete
from utils.utils import save_tb_scalars
from torch.utils.tensorboard import SummaryWriter

from algos.dqn.core import QNet


class Replay:
    def __init__(self, capacity=1e4):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    #     def load_replay(self):
    #         raise NotImplementedError

    def add_experience(self, experience):
        """
        Parameters:
        experience ([np.array]): Experience which should be stored in the replay buffer
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[int(self.position)] = experience
        self.position = (self.position + 1) % self.capacity

    def get_batch(self, batch_size):
        sample = random.sample(self.buffer, batch_size)

        obs_batch, act_batch, rew_batch, next_obs_batch, dones_batch = [], [], [], [], []

        for s in sample:
            obs_batch.append(s[0])
            act_batch.append(s[1])
            rew_batch.append(s[2])
            next_obs_batch.append(s[3])
            dones_batch.append(s[4])
        obs_batch = torch.tensor(obs_batch, dtype=torch.float32)
        act_batch = torch.tensor(act_batch, dtype=torch.int64)
        rew_batch = torch.tensor(rew_batch, dtype=torch.float32)
        next_obs_batch = torch.tensor(next_obs_batch, dtype=torch.float32)
        dones_batch = torch.tensor(dones_batch, dtype=torch.float32)
        return obs_batch, act_batch, rew_batch, next_obs_batch, dones_batch

    def get_size(self):
        return len(self.buffer)


class DQN:
    def __init__(self, name, env, seed=0,
                 replay_capacity=1e5, batch_size=100,
                 n_epochs=100, steps_per_epoch=400, start_steps=1000, update_after=1000,
                 gamma=0.99, tau=0.995, eps_start=0.9, eps_decay=200, eps_end=0.05, steps_target_update=100,
                 lr=1e-3, n_hidden_layers=2, n_neurons=64, batch_norm=False,
                 checkpoint_path=None, summary_path=None,
                 change_terminal_rew=False,
                 render=False, eval=False, resume=False,
                 **args):

        self.env = env

        self.action_space = env.action_space.n if isinstance(env.action_space, Discrete) \
            else env.action_space.shape[0]
        self.observation_space = env.observation_space.n if isinstance(env.observation_space, Discrete) \
            else env.observation_space.shape[0]

        if seed is not None:
            torch.manual_seed(seed)

        self.name = name

        # self.replay = Replay(replay_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.current_epoch = 0
        self.total_steps = 0
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.start_steps = start_steps
        self.update_after = update_after
        self.steps_target_update = steps_target_update

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        # initialize model and target model
        self.critic = QNet(self.env, n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, batch_norm=batch_norm)
        self.target_critic = QNet(self.env, n_hidden_layers=n_hidden_layers, n_neurons=n_neurons, batch_norm=batch_norm)
        for target_params in self.target_critic.parameters():
            target_params.requires_grad = False
        # copy parameter from critic to target_critic
        self._soft_update(tau=0)
        self.optimizer = Adam(self.critic.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        self.buffer = Replay(replay_capacity)


        self.eval = eval
        self.resume = resume
        self.render = render
        self.checkpoint_path = os.path.join(checkpoint_path, self.name + '.pt')
        self.summary_path = summary_path
        if not self.eval:
            self.writer = SummaryWriter(log_dir=self.summary_path)

        self.change_terminal_rew = change_terminal_rew

        if self.eval or self.resume:
            self.load_model(self.checkpoint_path)

    def update(self, data):
        obs_buff, act_buff, rew_buff, next_obs_buff, dones_buff = data
        act_buff = act_buff.unsqueeze(1)
        y = self.critic(obs_buff).gather(1, act_buff).squeeze()

        with torch.no_grad():
            next_state_action_values = torch.max(self.target_critic(next_obs_buff), 1).values.squeeze()
        target = rew_buff + self.gamma * (1 - dones_buff) * next_state_action_values

        # optimize Critic
        self.optimizer.zero_grad()
        loss = ((y - target) ** 2).mean()
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def train(self):    # play game against another agent
        # play game
        for epoch in range(self.current_epoch, self.n_epochs):
            self.current_epoch += 1
            if random.randint(0, 1):
                self.env.switch_side()
            obs = self.env.reset()
            obs = torch.tensor(obs).float()
            train_steps = 0
            reward_per_ep = []
            critic_per_ep = []
            bellman_per_ep = []
            diff_critic_bellman = []
            loss_critic_per_ep = []
            for t in range(self.steps_per_epoch):
                self.total_steps += 1
                train_steps += 1
                if train_steps < self.start_steps:
                    act = self.env.action_space.sample()
                else:
                    act = self.get_action(obs)

                next_obs, rew, done, _ = self.env.step(act)
                next_obs = torch.tensor(next_obs).float()
                if self.change_terminal_rew and done:
                    rew *= -1
                self.buffer.add_experience((obs.numpy(), act, rew, next_obs.numpy(), done))
                reward_per_ep.append(rew)

                # logging
                self.critic.eval()
                self.target_critic.eval()
                crit = self.critic.forward(obs.unsqueeze(0))[0][act].detach().item()
                bellm = rew + (1 - done) * self.gamma * torch.max(self.target_critic(next_obs.unsqueeze(0))).detach().item()
                critic_per_ep.append(crit)
                bellman_per_ep.append(bellm)
                diff_critic_bellman.append(crit - bellm)
                self.critic.train()
                self.target_critic.train()
                obs = next_obs

                if train_steps >= self.update_after:
                    data = self.buffer.get_batch(self.batch_size)
                    loss = self.update(data)

                    # logging
                    loss_critic_per_ep.append(loss)

                    # self._soft_update(self.tau)
                    if train_steps % self.steps_target_update == 0:
                        self._soft_update(tau=0)

                if done:
                    obs = self.env.reset()
                    obs = torch.tensor(obs).float()

            # logging
            if self.current_epoch % 10 == 0:
                mean_traj_rew = self.evaluate(10, render=False, print_reward=False)
                save_tb_scalars(self.writer,
                                self.total_steps,
                                reward_per_episode=torch.tensor(reward_per_ep, dtype=torch.float32).sum() / self.steps_per_epoch,
                                q_per_episode=torch.tensor(critic_per_ep).sum() / self.steps_per_epoch,
                                bellman_per_ep=torch.tensor(bellman_per_ep).sum() / self.steps_per_epoch,
                                diff_critic_bellman=torch.tensor(diff_critic_bellman).sum() / self.steps_per_epoch,
                                loss_q_per_ep=torch.stack(loss_critic_per_ep).sum() / self.steps_per_epoch,
                                epsilon=self.eps,
                                mean_traj_rew=mean_traj_rew
                                )
            else:
                save_tb_scalars(self.writer,
                                self.total_steps,
                                reward_per_episode=torch.tensor(reward_per_ep, dtype=torch.float32).sum() / self.steps_per_epoch,
                                q_per_episode=torch.tensor(critic_per_ep).sum() / self.steps_per_epoch,
                                bellman_per_ep=torch.tensor(bellman_per_ep).sum() / self.steps_per_epoch,
                                diff_critic_bellman=torch.tensor(diff_critic_bellman).sum() / self.steps_per_epoch,
                                loss_q_per_ep=torch.stack(loss_critic_per_ep).sum() / self.steps_per_epoch,
                                epsilon=torch.tensor(self.eps)
                                )

            self.save_model()
            self._update_eps()
        # Soft update
        # self._soft_update()

    def evaluate(self, n_episodes, render=None, print_reward=True):

        rend = render if render is not None else self.render

        cumulative_reward = 0
        traj_rewards = []
        traj_reward = 0
        step = 0
        episode = 0
        if random.randint(0, 1):
            self.env.switch_side()
        obs = self.env.reset()
        actions = []
        states = []
        print('Evaluating the deterministic policy...')
        while episode < n_episodes:
            step += 1
            action = self.get_action(torch.tensor(obs).float(), deterministic=True)
            states.append(obs)
            actions.append(action)
            new_obs, rew, done, _ = self.env.step(action)
            cumulative_reward += rew
            traj_reward += rew
            if rend:
                self.env.render()
            if done:
                step = 0
                episode += 1
                traj_rewards.append(traj_reward)
                traj_reward = 0
                if print_reward:
                    print(len(traj_rewards), 'trajectories: total', cumulative_reward, 'mean', np.mean(traj_rewards),
                          'std', np.std(traj_rewards),
                          'max', np.max(traj_rewards)
                          )
                states = []
                actions = []
                if random.randint(0, 1):
                    self.env.switch_side()
                obs = self.env.reset()
            else:
                obs = new_obs

        if print_reward:
            print('FINAL: total', cumulative_reward, 'mean', np.mean(traj_rewards), 'std', np.std(traj_rewards),
                  'max', np.max(traj_rewards))
        return cumulative_reward / n_episodes

    def get_action(self, obs, deterministic=False):
        """
        choose action based on epsilon-greedy.
        """
        if random.random() > self.eps or deterministic:
            # disallow all actions where a stone is already placed
            self.critic.eval()
            state_value = self.critic.forward(obs.unsqueeze(0))[0].detach()
            self.critic.train()
            for i in range(self.env.action_space.n):
                if obs[i] != 0:
                    state_value[i] = -1e7
            return state_value.argmax().item()

        else:
            return random.choice([a for a in range(self.env.action_space.n) if obs[a] == 0])

    def _soft_update(self, tau=None):
        polyak = tau if tau is not None else self.tau
        with torch.no_grad():
            for target_p, p in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_p.data.mul_(polyak)
                target_p.data.add_((1 - polyak) * p.data)

    def _update_eps(self):
        self.eps = self.eps_end * (1 - math.exp(-1 / self.eps_decay)) + self.eps * math.exp(-1 / self.eps_decay)

    def save_model(self, path=None):
        """
        saves current model to a given path
        :param path: (str) saving path for the model
        """
        checkpoint_path = path if path is not None else self.checkpoint_path
        data = {
            'epoch': self.current_epoch,
            'critic_state_dict': self.critic.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'critic_optim_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps
        }
        torch.save(data, checkpoint_path)

    def load_model(self, path=None):
        """
        loads a model from a given path
        :param path: (str) loading path for the model
        """
        load_path = path if path is not None else self.checkpoint_path
        checkpoint = torch.load(load_path)
        print('model loaded from', load_path)
        self.current_epoch = checkpoint['epoch']
        self.total_steps = checkpoint['total_steps']
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
