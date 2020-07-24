import torch
import torch.nn as nn
from gym.spaces import Discrete
import random


class QNet(nn.Module):
    def __init__(self, env, n_hidden_layers=2, n_neurons=64, batch_norm=False):
        super(QNet, self).__init__()

        self.action_space = env.action_space.n if isinstance(env.action_space, Discrete) \
            else env.action_space.shape[0]
        self.observation_space = env.observation_space.n if isinstance(env.observation_space, Discrete) \
            else env.observation_space.shape[0]

        sizes = [self.observation_space] + [n_neurons for _ in range(n_hidden_layers)] + [self.action_space]
        layers = []
        for i in range(n_hidden_layers + 1):
            if i < n_hidden_layers and batch_norm:
                layers.append(nn.BatchNorm1d(sizes[i]))
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < n_hidden_layers:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        return self.layers(obs)

    def get_action(self, observation, deterministic=False):
        obs = observation['board'] + observation['mark']
        obs = torch.tensor(obs).float()
        if random.random() > self.eps or deterministic:
            # disallow all actions where a stone is already placed
            self.critic.eval()
            state_value = self.critic.forward(obs.unsqueeze(0))[0].detach()
            self.critic.train()
            for i in range(self.env.action_space.n):
                if obs[i] != 0:
                    state_value[i] = -1e7
            return state_value.argmax().item()