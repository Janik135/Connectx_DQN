def agent(observation, configuration):
    import torch
    import torch.nn as nn
    from collections import OrderedDict
    from torch import tensor
    import random


    class QNet(nn.Module):
        def __init__(self, action_space, observation_space, n_hidden_layers=2, n_neurons=64, batch_norm=False):
            super(QNet, self).__init__()
            self.action_space = action_space
            self.observation_space = observation_space
    
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
    
        def get_action(self, observation):
            obs = observation['board'] + [observation['mark']]
            obs = torch.tensor(obs).float()
            self.eval()
            state_value = self.forward(obs.unsqueeze(0))[0].detach()
            self.train()
            for i in range(self.action_space):
                if obs[i] != 0:
                    state_value[i] = -1e7
            return state_value.argmax().item()
    action_space = 7
    observation_space = 43
    model = QNet(action_space, observation_space, batch_norm=True)
    model.eval()
    return int(model.get_action(observation))
