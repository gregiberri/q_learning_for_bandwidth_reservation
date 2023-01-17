from gym_environment import BREnv
from q_learning import test_baselines, test_network, load_dfs

_, no_df_val = load_dfs()

# exact booking
print('\nExact Booking')
booking = 'exactbooking'
test_env = BREnv(no_df_val, random_seed=1, under_ex_over_book_rates=[0, 1, 0], min_bs_number=10, max_bs_number=10)
test_baselines(test_env, result_folder=f'results/{booking}')
test_network(test_env, 'fc', network_path=f'results/{booking}/double_dqn.pth', result_path=f'results/{booking}/double_dqn_policy.csv,old.old')
test_network(test_env, 'fc', network_path=f'results/{booking}/dqn.pth', result_path=f'results/{booking}/dqn_policy.csv')
test_network(test_env, 'dueling_fc', network_path=f'results/{booking}/dueling_dqn.pth', result_path=f'results/{booking}/dueling_dqn_policy.csv')
#
# # underbooking
print('\nUnder Booking')
booking = 'underbooking'
test_env = BREnv(no_df_val, random_seed=1, under_ex_over_book_rates=[1, 0, 0], min_bs_number=10, max_bs_number=10)
test_baselines(test_env, result_folder=f'results/{booking}')
test_network(test_env, 'fc', network_path=f'results/{booking}/double_dqn.pth', result_path=f'results/{booking}/double_dqn_policy.csv,old.old')
test_network(test_env, 'fc', network_path=f'results/{booking}/dqn.pth', result_path=f'results/{booking}/dqn_policy.csv')
test_network(test_env, 'dueling_fc', network_path=f'results/{booking}/dueling_dqn.pth', result_path=f'results/{booking}/dueling_dqn_policy.csv')

# overbooking
print('\nOver Booking')
booking = 'overbooking'
test_env = BREnv(no_df_val, random_seed=1, under_ex_over_book_rates=[0, 0, 1], min_bs_number=10, max_bs_number=10)
test_baselines(test_env, result_folder=f'results/{booking}')
test_network(test_env, 'fc', network_path=f'results/{booking}/double_dqn.pth', result_path=f'results/{booking}/double_dqn_policy.csv,old.old')
test_network(test_env, 'fc', network_path=f'results/{booking}/dqn.pth', result_path=f'results/{booking}/dqn_policy.csv')
test_network(test_env, 'dueling_fc', network_path=f'results/{booking}/dueling_dqn.pth', result_path=f'results/{booking}/dueling_dqn_policy.csv')

# mixed booking
print('\nMixed Booking')
booking = 'mixedbooking'
test_env = BREnv(no_df_val, random_seed=1, under_ex_over_book_rates=[1 / 3, 1 / 3, 1 / 3], min_bs_number=10, max_bs_number=10)
test_baselines(test_env, result_folder=f'results/{booking}')
test_network(test_env, 'fc', network_path=f'results/{booking}/double_dqn.pth', result_path=f'results/{booking}/double_dqn_policy.csv,old.old')
test_network(test_env, 'fc', network_path=f'results/{booking}/dqn.pth', result_path=f'results/{booking}/dqn_policy.csv')
test_network(test_env, 'dueling_fc', network_path=f'results/{booking}/dueling_dqn.pth', result_path=f'results/{booking}/dueling_dqn_policy.csv')
