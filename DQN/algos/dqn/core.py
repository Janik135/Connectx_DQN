import torch.nn as nn
from gym.spaces import Discrete


class QNet(nn.Module):
    def __init__(self, env, n_hidden_layers=2, n_neurons=64):
        super(QNet, self).__init__()

        self.action_space = env.action_space.n if isinstance(env.action_space, Discrete) \
            else env.action_space.shape[0]
        self.observation_space = env.observation_space.n if isinstance(env.observation_space, Discrete) \
            else env.observation_space.shape[0]

        sizes = [self.observation_space] + [n_neurons for _ in range(n_hidden_layers)] + [self.action_space]
        layers = []
        for i in range(n_hidden_layers + 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < n_hidden_layers - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        return self.layers(obs)
