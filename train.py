from q_learning import train

train(network='fc',
      save_path='results/mixedbooking/double_dqn.pth',
      under_ex_over_book_rates=[1 / 3, 1 / 3, 1 / 3])

train(network='fc',
      save_path='results/exactbooking/dqn.pth',
      under_ex_over_book_rates=[0, 1, 0])

train(network='fc',
      save_path='results/underbooking/double_dqn.pth',
      under_ex_over_book_rates=[1, 0, 0])

train(network='fc',
      save_path='results/overbooking/dqn.pth',
      under_ex_over_book_rates=[0, 0, 1])
