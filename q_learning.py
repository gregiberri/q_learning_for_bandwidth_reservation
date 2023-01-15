import os

import gym
import numpy as np
import pandas as pd
import torch
from tianshou.env import DummyVectorEnv
import tianshou as ts

from classical_policies import NoPolicyNet, GreedyPolicyNet
from data_preprocess import load_and_preproces_datasets, train_val_split
from gym_environment import BREnv, MAX_BS_LENGTH
from tianshou.utils.net.common import Net, Recurrent


def select_network(state_shape: np.array, action_shape: np.array, network: str = 'fc', device: str = 'cpu') -> torch.nn.Module:
    """
    Select the network.

    :param state_shape: the shape of the state space
    :param action_shape: the shape of the action space
    :param network: string of the network type: `fc`, `dueling_fc`, `rnn`, `no_policy`, `greedy`
    :param device: whether to use `cpu` or `cuda` device

    :return: the net network
    """
    if network == 'fc':
        return Net(state_shape,
                   action_shape,
                   hidden_sizes=[128, 128, 128, 128],
                   device=device,
                   )
    elif network == 'dueling_fc':
        V_param = Q_param = {'input_dim': state_shape, 'output_dim': action_shape, 'hidden_sizes': [128, 128]}
        return Net(state_shape,
                   action_shape,
                   hidden_sizes=[128, 128, 128, 128],
                   device=device,
                   dueling_param=(Q_param, V_param),
                   )
    elif network == 'rnn':
        return Recurrent(layer_num=3,
                         state_shape=state_shape,
                         action_shape=action_shape,
                         device=device
                         )
    elif network == 'no_policy':
        return NoPolicyNet(state_shape=state_shape,
                           action_shape=action_shape)
    elif network == 'greedy':
        return GreedyPolicyNet(state_shape=state_shape,
                               action_shape=action_shape)
    else:
        raise ValueError(f'Wrong value for network: {network}')


def evaluate_network(net, env: gym.Env):
    """
    Evaluate the network

    :param net:
    :param env:
    :return:
    """
    rewards = []
    observations = []
    actions = []

    done = False
    obs = env.reset()
    while not done:
        act = torch.argmax(net(torch.tensor(obs).view(1, -1))[0])
        obs_next, rew, done, info = env.step(act)

        # record the event
        actions.append(int(act))
        rewards.append(rew)
        observations.append(obs)

        obs = obs_next

    # make a dict of the observations
    observations = np.array(observations).T.tolist()
    obs_dict = dict(zip(['current_price', 'no1_price', 'no2_price', 'no1_future_price', 'no2_future_price', 'bookingtype', 'timesteps_left'],
                        observations))
    obs_dict.update({'actions': actions, 'rewards': rewards})
    # make a pandas dataframe of the state, reward, observations
    df = pd.DataFrame(obs_dict)

    return df


def load_dfs():
    no_df = load_and_preproces_datasets()
    no_df_train, no_df_val = train_val_split(no_df)
    return no_df_train.reset_index(drop=True), no_df_val.reset_index(drop=True)


def make_envs():
    no_df_train, no_df_val = load_dfs()

    train_env = DummyVectorEnv([lambda: BREnv(no_df_train) for _ in range(10)])
    test_env = DummyVectorEnv([lambda: BREnv(no_df_val, random_seed=1) for i in range(1)])

    return train_env, test_env


def get_env_state_and_action_spaces(train_env):
    state_shape = train_env.workers[0].env.observation_space.shape or train_env.workers[0].env.observation_space.n
    action_shape = train_env.workers[0].env.action_space.shape or train_env.workers[0].env.action_space.n

    return state_shape, action_shape


def train(network='fc',
          lr=1e-3,
          weight_decay=1e-6,
          discount_factor=0.99,
          estimation_step=3,
          batch_size=128,
          is_double=True,
          save_path='results/dqn.pth'):
    # make the envs
    train_env, test_env = make_envs()

    # get the state and action shapes
    state_shape, action_shape = get_env_state_and_action_spaces(train_env)

    # model, optimizer and policy
    net = select_network(state_shape, action_shape, network=network)
    optim = torch.optim.Adam(net.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)
    policy = ts.policy.DQNPolicy(net, optim,
                                 discount_factor=discount_factor,
                                 estimation_step=estimation_step,
                                 target_update_freq=300,
                                 is_double=is_double)

    # make collectors to collect episodes
    train_collector = ts.data.Collector(policy, train_env, ts.data.VectorReplayBuffer(20000, len(train_env)), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env, ts.data.VectorReplayBuffer(20000, len(test_env)))

    # train
    print('Start training.')
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=20, step_per_epoch=1000 * MAX_BS_LENGTH, step_per_collect=100,
        update_per_step=0.1, episode_per_test=1, batch_size=batch_size,
        train_fn=lambda epoch, env_step: policy.set_eps(0.5),
        test_fn=lambda epoch, env_step: policy.set_eps(0.0),
        save_best_fn=lambda policy: torch.save(net.state_dict(), save_path),
        verbose=True, show_progress=True, test_in_train=True
    )
    print(f'Finished training! Use {result["best_result"]}')


def test_baselines(env, result_folder: str = 'results'):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # make and evaluate no_policy
    net = select_network(state_shape, action_shape, network='no_policy')
    no_policy_df = evaluate_network(net, env)
    print(f'No net reward: {no_policy_df.rewards.sum()}')
    no_policy_df.to_csv(os.path.join(result_folder, 'no_policy.csv'), index=False)

    # make and evaluate greedy net
    net = select_network(state_shape, action_shape, network='greedy')
    greedy_policy_df = evaluate_network(net, env)
    print(f'Greedy net reward: {greedy_policy_df.rewards.sum()}')
    greedy_policy_df.to_csv(os.path.join(result_folder, 'greedy_policy.csv'), index=False)


def test_network(env, network_type, network_path: str = 'results/dqn.pth', result_path: str = 'results/dqn_policy.csv'):
    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    # model
    net = select_network(state_shape, action_shape, network=network_type)
    net.load_state_dict(torch.load(network_path))
    policy_df = evaluate_network(net, env)
    print(f'Trained net reward: {policy_df.rewards.sum()}')

    policy_df.to_csv(result_path, index=False)
