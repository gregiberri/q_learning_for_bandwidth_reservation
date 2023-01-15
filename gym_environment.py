import re

import pandas as pd
import gym
from gym import spaces
import numpy as np
import random

from data_preprocess import load_and_preproces_datasets, train_val_split, PRICE_RESOLUTION

MIN_BS_LENGTH_MINUTE = 10
MAX_BS_LENGTH_MINUTE = 20
MIN_BS_NUMBER = 3
MAX_BS_NUMBER = 10
MIN_UNOV_BOOK = 0.5
MAX_UNOV_BOOK = 5
UNDER_EX_OVER_BOOK_RATES = [1, 0, 0]
CANCELATION_FEE_PER_MINUTE = 0.1  # per minutes

MIN_BS_LENGTH = int(MIN_BS_LENGTH_MINUTE * (60 / PRICE_RESOLUTION))
MAX_BS_LENGTH = int(MAX_BS_LENGTH_MINUTE * (60 / PRICE_RESOLUTION))
CANCELATION_FEE = CANCELATION_FEE_PER_MINUTE / (60 / PRICE_RESOLUTION)
CONSTRAINT_VIOLATION_COST = 1

PRERESERVE_TIME = 100


class BREnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 no_df: pd.DataFrame,
                 min_bs_length=MIN_BS_LENGTH,
                 max_bs_length=MAX_BS_LENGTH,
                 min_bs_number=MIN_BS_NUMBER,
                 max_bs_number=MAX_BS_NUMBER,
                 min_unov_book=MIN_UNOV_BOOK,
                 max_unov_book=MAX_UNOV_BOOK,
                 under_ex_over_book_rates=UNDER_EX_OVER_BOOK_RATES,
                 random_seed: int = None
                 ):
        """

        Args:
            no_df:
            min_bs_length:
            max_bs_length:
            min_bs_number:
            max_bs_number:
            min_unov_book:
            max_unov_book:
            under_ex_over_book_rates:
            random_seed:
        """
        super(BREnv, self).__init__()
        # get the number of columns named no*
        r = re.compile("no.*")
        self.no_number = len(list(filter(r.match, no_df.columns)))

        # Define action and observation space
        # Discrete actions
        self.action_space = spaces.Discrete(2 * self.no_number + 1)
        # The observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 2 * self.no_number + 3), dtype=np.float16)

        # Reward range
        # self.reward_range = [-1, -0.3] todo: delete?
        self.no_df = no_df

        self.min_bs_length = min_bs_length
        self.max_bs_length = max_bs_length
        self.min_bs_number = min_bs_number
        self.max_bs_number = max_bs_number
        self.min_unov_book = min_unov_book
        self.max_unov_book = max_unov_book
        self.under_ex_over_book_rates = under_ex_over_book_rates
        self.random_seed = random_seed

    def set_random_seed(self):
        """
        Set the random seed: it should be None during training: hence all reset will yield different random numbers,
        but should be a number for validation: hence the random numbers will be the same for all validation episode
        """
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def reset(self):
        self.set_random_seed()

        # Reset the state of the environment to an initial state
        self.current_bs = 0
        self.current_bs_step = 0

        # Set the episode length
        self.bs_number = random.randint(self.min_bs_number, self.max_bs_number)
        self.bs_starts = np.random.randint(self.max_bs_number * self.max_bs_length + PRERESERVE_TIME,
                                           len(self.no_df) - self.max_bs_number * self.max_bs_length + PRERESERVE_TIME,
                                           size=self.bs_number)
        self.bs_lengths = np.random.randint(self.min_bs_length, self.max_bs_length, size=self.bs_number)

        # Get for all the BS the current and the next prices
        self.bs_prices = []
        self.bs_future_prices = []
        for i in range(len(self.bs_lengths)):
            self.bs_prices.append(
                self.no_df.loc[self.bs_starts[i]:self.bs_starts[i] + self.bs_lengths[i] - 1].reset_index())
            if i == len(self.bs_lengths) - 1:  # last
                random_start = random.randint(self.bs_lengths[i], len(self.no_df))
                self.bs_future_prices.append(
                    self.no_df.loc[random_start - self.bs_lengths[i]:random_start - 1].reset_index())
            else:
                self.bs_future_prices.append(
                    self.no_df.loc[self.bs_starts[i + 1] - self.bs_lengths[i]:self.bs_starts[i + 1] - 1].reset_index())

        # Get the index at the beginning at t0
        timesteps_from_start = np.cumsum(self.bs_lengths) - self.bs_lengths
        prereserved_indices = [bs_start - timestep_from_start - PRERESERVE_TIME
                               for bs_start, timestep_from_start in zip(self.bs_starts, timesteps_from_start)]

        # Prereserved prices
        prereserved_prices = np.min(self.no_df.loc[prereserved_indices][['no1', 'no2']], axis=1).to_list()
        self.prereserved_prices = [[price] * bs_length for price, bs_length in zip(prereserved_prices, self.bs_lengths)]
        self.last_price = self.prereserved_prices[0][0]

        # Time till the next BS
        self.timestep_left_from_bs = self.bs_lengths[0]

        # Set the BSs to under, exact or overbooking
        self.exunov_bookings = np.random.choice([-1, 0, 1], size=self.bs_number, p=self.under_ex_over_book_rates)
        self.exunov_times = []
        for exunov_booking in self.exunov_bookings:
            if exunov_booking in [-1, 1]:  # under- or overbooking
                self.exunov_times.append(random.randint(MIN_UNOV_BOOK * (60 / PRICE_RESOLUTION),
                                                        MAX_UNOV_BOOK * (60 / PRICE_RESOLUTION)))
            else:
                self.exunov_times.append(0)

        return self._get_obs()

    def _get_obs(self):
        # get the current available prices
        self.current_prices = self.bs_prices[self.current_bs].loc[self.current_bs_step][['no1', 'no2']].to_list()
        self.future_prices = self.bs_future_prices[self.current_bs].loc[self.current_bs_step][['no1', 'no2']].to_list()
        under_or_over = self.exunov_bookings[self.current_bs]

        return [self.last_price] + self.current_prices + self.future_prices + [under_or_over, self.timestep_left_from_bs / MAX_BS_LENGTH]

    def step(self, action):
        cost = self._take_action(action)
        reward = - cost

        # step to next
        self.current_bs_step += 1
        self.timestep_left_from_bs -= 1

        # If we arrived to a next bs reset everything to the next bs
        if self.timestep_left_from_bs == 0:
            # If it was the last bs`s last step we terminate the episode, only until the last-1 because we need one bs ahead for future bs price, maybe later new idea
            if self.current_bs == self.bs_number - 1:
                self.current_bs_step -= 1
                # todo big negative reward if under or overbook, but it was not done
                return self._get_obs(), reward, True, {}

            self.current_bs += 1
            self.current_bs_step = 0
            self.last_price = self.prereserved_prices[self.current_bs][0]
            self.timestep_left_from_bs = self.bs_lengths[self.current_bs]

        obs = self._get_obs()

        return obs, reward, False, {}

    def _take_action(self, action):
        if self.random_seed is not None:
            if self.last_price != np.min(self.current_prices):
                asd = 1  # todo delete

        # we pay the reserved price for the current timestep
        cost = self.last_price

        # do nothing
        if action == 0:
            return cost

        # pay the cancelation fee and update the reservation
        elif action in [1, 2]:
            cost += CANCELATION_FEE * self.timestep_left_from_bs
            self.last_price = self.current_prices[action - 1]

        # solve the under or overbooking
        elif action in [3, 4]:
            if self.exunov_bookings[self.current_bs] == 0:  # exact booking, but we would like to solve the underbooking scenario
                cost += CONSTRAINT_VIOLATION_COST
                return cost

            else:
                # underbooking: the new price we prereserve on is one of the prices of the current NOs
                if self.exunov_bookings[self.current_bs] == -1:
                    # we pay the fee for canceling the underbooked time in the next BS, besides the last BS (there is nothing to cancel)
                    cost += CANCELATION_FEE * self.exunov_times[self.current_bs] if self.current_bs < self.bs_number else 0

                    # select the price from the current NOs
                    new_prereserve_price = self.current_prices[action - self.no_number - 1]

                    # make the changes in the variables for solving underbooking
                    self.solve_underbooking(new_prereserve_price)

                # overbooking: the new price we prereserve on is one of the prices of the future NOs
                elif self.exunov_bookings[self.current_bs] == 1:
                    # we pay the fee for canceling the overbooked time in the current BS
                    cost += CANCELATION_FEE * self.exunov_times[self.current_bs]

                    # select the price from the next NOs
                    new_prereserve_price = self.future_prices[action - self.no_number - 1]

                    # make the changes in the variables for solving overbooking
                    self.solve_overbooking(new_prereserve_price)

                # the under- or overbooking is solved, so there is exact booking for the current BS now
                self.exunov_bookings[self.current_bs] = 0

        return cost

    def solve_underbooking(self, new_prereserve_price):
        # we add the underbooked amount to the time left from the current BS
        self.timestep_left_from_bs += self.exunov_times[self.current_bs]

        # we add the underbooked amount to the length of the current BS, and subtract it from the next BS
        self.bs_lengths[self.current_bs] += self.exunov_times[self.current_bs]
        self.bs_lengths[self.current_bs + 1] -= self.exunov_times[self.current_bs]

        # set the prices to the underbooking
        # the prices of the underbooked timeperiod is appended to the current BS's prices
        self.bs_prices[self.current_bs] = self.no_df.loc[self.bs_prices[self.current_bs]['index'].iloc[0]:
                                                         self.bs_prices[self.current_bs]['index'].iloc[-1] +
                                                         self.exunov_times[self.current_bs]].reset_index()
        # the prices of the underbooked timeperiod is taken from to the next BS's prices
        self.bs_prices[self.current_bs + 1] = self.no_df.loc[self.bs_prices[self.current_bs + 1]['index'].iloc[0] +
                                                             self.exunov_times[self.current_bs]:
                                                             self.bs_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

        # the prices of the underbooked timeperiod is appended to the future prices in the current BS
        self.bs_future_prices[self.current_bs] = self.no_df.loc[self.bs_future_prices[self.current_bs]['index'].iloc[0]:
                                                                self.bs_future_prices[self.current_bs]['index'].iloc[-1] +
                                                                self.exunov_times[self.current_bs]].reset_index()
        # the prices of the underbooked timeperiod is taken from the future prices in the next BS
        self.bs_future_prices[self.current_bs + 1] = self.no_df.loc[self.bs_future_prices[self.current_bs + 1]['index'].iloc[0] +
                                                                    self.exunov_times[self.current_bs]:
                                                                    self.bs_future_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

        # Update the prereserved prices
        # append the currently available price for the underbooked time to the current BS's prereserved prices
        self.prereserved_prices[self.current_bs] = self.prereserved_prices[self.current_bs] + [new_prereserve_price] * self.exunov_times[self.current_bs]
        # take the prereserved prices for the underbooked time from the next BS
        self.prereserved_prices[self.current_bs + 1] = self.prereserved_prices[self.current_bs + 1][self.exunov_times[self.current_bs]:]

    def solve_overbooking(self, new_prereserve_price):
        # we subtract the overbooked amount from the time left from the current BS
        self.timestep_left_from_bs -= self.exunov_times[self.current_bs]

        # we subtract the overbooked amount form the length of the current BS, and add it to the next BS
        self.bs_lengths[self.current_bs] -= self.exunov_times[self.current_bs]
        self.bs_lengths[self.current_bs + 1] += self.exunov_times[self.current_bs]

        # set the prices to the overbooking
        # the prices of the overbooked timeperiod is taken from the current BS's prices
        self.bs_prices[self.current_bs] = self.no_df.loc[self.bs_prices[self.current_bs]['index'].iloc[0]:
                                                         self.bs_prices[self.current_bs]['index'].iloc[-1] -
                                                         self.exunov_times[self.current_bs]].reset_index()
        # the prices of the overbooked timeperiod appended to the next BS's prices
        self.bs_prices[self.current_bs + 1] = self.no_df.loc[self.bs_prices[self.current_bs + 1]['index'].iloc[0] -
                                                             self.exunov_times[self.current_bs]:
                                                             self.bs_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

        # the prices of the overbooked timeperiod is taken from the future prices in the current BS
        self.bs_future_prices[self.current_bs] = self.no_df.loc[self.bs_future_prices[self.current_bs]['index'].iloc[0]:
                                                                self.bs_future_prices[self.current_bs]['index'].iloc[-1] -
                                                                self.exunov_times[self.current_bs]].reset_index()
        # the prices of the overbooked timeperiod is added to the future prices in the next BS
        self.bs_future_prices[self.current_bs + 1] = self.no_df.loc[self.bs_future_prices[self.current_bs + 1]['index'].iloc[0] -
                                                                    self.exunov_times[self.current_bs]:
                                                                    self.bs_future_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

        # Update the prereserved prices
        # take the prereserved prices for the overbooked time from the current BS
        self.prereserved_prices[self.current_bs] = self.prereserved_prices[self.current_bs][:-self.exunov_times[self.current_bs]]
        # append the currently available price for the overbooked time to the next BS's prereserved prices
        self.prereserved_prices[self.current_bs + 1] = [new_prereserve_price] * self.exunov_times[self.current_bs] + self.prereserved_prices[self.current_bs + 1]


if __name__ == '__main__':
    no_df = load_and_preproces_datasets()
    no_df_train, no_df_val = train_val_split(no_df)
    env = BREnv(no_df_train)
    env.reset()
    env.step(4)
    env.step(1)
    env.step(1)
