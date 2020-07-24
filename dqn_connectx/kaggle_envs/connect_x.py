import gym
from kaggle_environments import make


class ConnectX(gym.Env):
    def __init__(self):
        super(ConnectX, self).__init__()
        self.env = make('connectx', debug=False)
        self.pair = [None, 'random']
        self.side = 0
        self.trainer = self.env.train(self.pair)

        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows + 1)

    def step(self, action):
        obs, rew, done, info = self.trainer.step(action)
        obs = obs['board'] + [obs['mark']]

        if done and rew is None:
            rew = 0
        # if done:
        #     if rew == 1:
        #         rew = 20
        #     elif rew == -1:
        #         rew = -20
        #     else:
        #         rew = 10
        # else:
        #     rew = 0.5

        return obs, rew, done, info

    def reset(self):
        obs = self.trainer.reset()
        return obs['board'] + [obs['mark']]

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def change_opponent(self, agent2):
        self.pair = [None, agent2]

    def get_side(self):
        return self.side

    def switch_side(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)
        self.side = 1 - self.side % 2
