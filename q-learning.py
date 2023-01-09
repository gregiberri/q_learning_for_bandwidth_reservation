import gym
import torch, numpy as np
from torch import nn
import tianshou as ts

from data_preprocess import load_and_preproces_datasets, train_val_split
from gym_environment import BREnv


class Net(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 128), nn.ReLU(inplace=True),
            nn.Linear(128, np.prod(action_shape)),
        )

    def forward(self, obs, state=None, info={}):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


if __name__ == '__main__':
    no_df = load_and_preproces_datasets()
    no_df_train, no_df_val = train_val_split(no_df)
    no_df_train, no_df_val = no_df_train.reset_index(drop=True), no_df_val.reset_index(drop=True)
    env = BREnv(no_df_train)
    test_env = BREnv(no_df_val)

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net = Net(state_shape, action_shape)
    optim = torch.optim.Adam(net.parameters(), lr=1e-3)

    policy = ts.policy.DQNPolicy(net, optim, discount_factor=0.9, estimation_step=3, target_update_freq=300)

    train_collector = ts.data.Collector(policy, env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_env, ts.data.VectorReplayBuffer(20000, 10), exploration_noise=True)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=10, step_per_epoch=10000, step_per_collect=10,
        update_per_step=0.1, episode_per_test=100, batch_size=2,
        train_fn=lambda epoch, env_step: policy.set_eps(0.5),
        test_fn=lambda epoch, env_step: policy.set_eps(0.05),
        verbose=True, show_progress=True, test_in_train=False)
    print(f'Finished training! Use {result["duration"]}')
