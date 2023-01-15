from gym_environment import BREnv
from q_learning import test_baselines, test_network, load_dfs

_, no_df_val = load_dfs()
test_env = BREnv(no_df_val, random_seed=1)
test_baselines(test_env)
test_network(test_env, 'fc', network_path='results/double_dqn.pth', result_path='results/double_dqn_policy.csv')
