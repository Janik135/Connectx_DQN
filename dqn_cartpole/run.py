import argparse
import os
from pprint import pprint

import gym
import yaml

from algos.dqn.dqn import DQN

from utils.utils import unique_path


def _parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Solve the different gym environments')
    parser.add_argument('--name', type=str, required=True, help='identifier to store experiment results')
    parser.add_argument('--env', type=str, required=True, help='name of the environment to be learned')

    parser.add_argument('--seed', type=int, help='seed for torch/numpy/gym to make experiments reproducible')

    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--experiment', action='store_true', help='whether this experiment was run via experiments.py')

    parser.add_argument('--n_eval_traj', type=int, default=25,
                        help='number of trajectories to run evaluation on, when --eval is set.')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--eval', action='store_true', help='toggles evaluation mode')
    group.add_argument('--resume', action='store_true',
                       help='resume training on an existing model by loading the last checkpoint')

    subparsers = parser.add_subparsers(description='Algorithms to choose from', dest='command')

    # ---------------------
    #  dqn_cartpole Arguments
    # ---------------------
    dqn_parser = subparsers.add_parser('dqn_cartpole')
    dqn_parser.add_argument('--n_epochs', type=int, help='number of training epochs')
    dqn_parser.add_argument('--n_steps', type=int, help='number of environment steps per epoch')
    # hyper-parameter
    dqn_parser.add_argument('--gamma', type=float, help='1 minus environment reset probability.')
    dqn_parser.add_argument('--tau', type=float, help='Polyak Averaging')
    dqn_parser.add_argument('--lr', type=float, help='Learning rate of the Q-function')
    dqn_parser.add_argument('--batch_size', type=float, help='Size of a mini-batch')
    dqn_parser.add_argument('--replay_capacity', type=float, help='Samples stored in replay buffer')

    args = parser.parse_args()
    if args.command is None:
        raise Exception('No algorithm specified! The first argument')

    args.algo = args.command
    del args.command
    return args


def run_single_experiment(args=None):
    if args is None:
        args = _parse()
        default_path = os.path.join('hyperparameters', args.algo.lower(), f"{args.env}.yaml")
        print(default_path)
        params = {}
        if os.path.exists(default_path):
            with open(default_path) as f:
                params = yaml.load(f, Loader=yaml.FullLoader)['params']
        print(params)

        if args.experiment:
            args.summary_path = os.path.join('out', 'experiments', '_'.join(args.name.split('_')[:-1]), 'summary',
                                             args.name)
            args.checkpoint_path = os.path.join('out', 'experiments', '_'.join(args.name.split('_')[:-1]), 'models',
                                                args.name)
        elif args.resume or args.eval:
            args.summary_path = os.path.join('out', args.name, 'summary')
            args.checkpoint_path = os.path.join('out', args.name, 'models')
        else:
            args.name = unique_path('out', args.name)
            args.summary_path = os.path.join('out', args.name, 'summary')
            args.checkpoint_path = os.path.join('out', args.name, 'models')
            # make sure the necessary directories exist
            # if not args.eval:
            os.makedirs(args.summary_path, exist_ok=True)
            os.makedirs(args.checkpoint_path, exist_ok=True)

        args = vars(args)
        for k, v in args.items():
            if v is not None:
                params[k] = v
        params['seed'] = args['seed']  # has to be set extra, because seed=None is a valid option
        args = params

    print('ARGUMENTS')
    pprint(args)
    print()

    args['eval'] = args['eval'] if 'eval' in args else False
    args['resume'] = args['resume'] if 'resume' in args else False
    args['seed'] = args['seed'] if 'seed' in args else None

    if not args['resume'] and not args['eval']:
        # save arguments
        params_path = os.path.join(args['checkpoint_path'], 'params.yaml')
        with open(params_path, 'w') as outfile:
            yaml.dump({'params': args}, outfile, default_flow_style=None)
            print('saved args! location:', os.path.join(args['checkpoint_path'], 'params.yaml'))

    # create gym environment
    args['env'] = gym.make(args['env'])

    # seed gym environment if given
    if args['seed'] is not None:
        args['env'].seed(args["seed"])
        print(f'# Seeded Env. Seed={args["seed"]}')

    # select, instantiate and train correct algorithm
    model = {'dqn_cartpole': DQN}[args['algo'].upper()](**args)
    print(args['seed'])
    print(model)
    if 'eval' in args and args['eval']:
        model.evaluate(args['n_eval_traj'], print_reward=True)
    else:
        model.train()
    args['env'].close()


if __name__ == '__main__':
    run_single_experiment()
