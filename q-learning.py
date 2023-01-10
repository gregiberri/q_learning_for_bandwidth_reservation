import torch
from gym.wrappers import TimeLimit
from tianshou.env import DummyVectorEnv
import tianshou as ts

from data_preprocess import load_and_preproces_datasets, train_val_split
from gym_environment import BREnv, MAX_BS_LENGTH
from tianshou.utils.net.common import Net

if __name__ == '__main__':
    no_df = load_and_preproces_datasets()
    no_df_train, no_df_val = train_val_split(no_df)
    no_df_train, no_df_val = no_df_train.reset_index(drop=True), no_df_val.reset_index(drop=True)
    env = TimeLimit(BREnv(no_df_train))
    train_env = DummyVectorEnv([lambda: TimeLimit(BREnv(no_df_train), 15000) for _ in range(10)])
    test_env = DummyVectorEnv([lambda: TimeLimit(BREnv(no_df_val, random_seed=i), 15000) for i in range(1)])

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    # model
    net = Net(state_shape,
              action_shape,
              hidden_sizes=[128, 128, 128, 128],
              device='cuda',
              # dueling=(Q_param, V_param),
              ).cuda()
    optim = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=300)

    train_collector = ts.data.Collector(policy, train_env, ts.data.VectorReplayBuffer(20000, len(train_env)),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env, ts.data.VectorReplayBuffer(20000, len(test_env)),
                                       exploration_noise=True)

    print('Start training.')
    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=30, step_per_epoch=1000 * MAX_BS_LENGTH, step_per_collect=100,
        update_per_step=0.1, episode_per_test=1, batch_size=128,
        train_fn=lambda epoch, env_step: policy.set_eps(0.5),
        test_fn=lambda epoch, env_step: policy.set_eps(0.0),
        verbose=True, show_progress=True, test_in_train=True
    )
    print(f'Finished training! Use {result["duration"]}')
