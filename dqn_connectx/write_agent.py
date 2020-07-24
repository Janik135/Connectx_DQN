import os
import argparse
import inspect
import sys
import torch
from submission.connectx.model import QNet
from kaggle_envs.connect_x import ConnectX
from kaggle_environments import make, utils


def _parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Argparser to work with kaggles environments')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    # parser.add_argument('--eval', action='store_true', help='evaluates your agent against ')
    parser.add_argument('--write', action='store_true', help='writes a submission file for your agent')
    args = parser.parse_args()
    return args


def write_agent_to_file(env, save_file, model_class, load_path):
    checkpoint = torch.load(load_path)
    action_space = env.action_space.n
    observation_space = env.observation_space.n
    state_dict = checkpoint['critic_state_dict']

    with open(save_file, 'w') as f:
        f.write('def agent(observation, configuration):\n'
                '    import torch\n' +
                '    import torch.nn as nn\n' +
                '    from collections import OrderedDict\n' +
                # '    from gym.spaces import Discrete\n' +
                '    from torch import tensor\n' +
                '    import random\n\n\n')
        model_source = inspect.getsourcelines(model_class)
        for line in model_source[0]:
            f.write("    {}".format(line))
        # f.write(f'    state_dict = {state_dict}\n') # TODO: without loading the state dict the params are initialized randomly
        f.write(f'    action_space = {action_space}\n' +
                f'    observation_space = {observation_space}\n' +
                '    model = QNet(action_space, observation_space, batch_norm=True)\n' +
                # '    model.load_state_dict(state_dict)\n' +  # TODO: see above
                '    model.eval()\n' +
                '    return int(model.get_action(observation))\n')
    print("model written to", save_file)


if __name__ == '__main__':
    args = _parse()
    if args.write:
        env = {'connectx': ConnectX}[args.env]()
        file = os.path.join('submission', args.env, 'submission.py')
        load_path = os.path.join('out', args.name, 'models', args.name + '.pt')
        write_agent_to_file(env, file, QNet, load_path)

        out = sys.stdout
        submission = utils.read_file(file)
        agent = utils.get_last_callable(submission)
        sys.stdout = out

        env = make(args.env, debug=True)
        env.run([agent, agent])
        print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")
