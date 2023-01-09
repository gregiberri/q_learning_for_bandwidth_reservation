import pandas as pd
import os
import tqdm
import datetime as dt
import gym
from gym import spaces
import numpy as np
import random

from data_preprocess import load_and_preproces_datasets, train_val_split, PRICE_RESOLUTION

MIN_BS_LENGTH = 600
MAX_BS_LENGTH = 1200
MIN_BS_NUMBER = 3
MAX_BS_NUMBER = 10
MIN_UNOV_BOOK = 0.5
MAX_UNOV_BOOK = 5
UNDER_EX_OVER_BOOK_RATES = [0, 1, 0]

CANCELATION_FEE_PER_MINUTE = 0.00010  # per minutes
CANCELATION_FEE = CANCELATION_FEE_PER_MINUTE / (60 / PRICE_RESOLUTION)
CONSTRAINT_VIOLATION_COST = 1000

PRERESERVE_TIME = 1


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
                 under_ex_over_book_rates=UNDER_EX_OVER_BOOK_RATES
                 ):
        super(BREnv, self).__init__()
        # Define action and observation space
        # Discrete actions
        self.action_space = spaces.Discrete(7)
        # The observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(1, 9), dtype=np.float16)

        # Reward range
        # self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.no_df = no_df

        self.min_bs_length = min_bs_length
        self.max_bs_length = max_bs_length
        self.min_bs_number = min_bs_number
        self.max_bs_number = max_bs_number
        self.min_unov_book = min_unov_book
        self.max_unov_book = max_unov_book
        self.under_ex_over_book_rates = under_ex_over_book_rates

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_bs = 0
        self.current_bs_step = 0

        # Set the episode length
        self.bs_number = random.randint(self.min_bs_number, self.max_bs_number)
        self.bs_starts = np.random.randint(self.max_bs_number * self.max_bs_length + PRERESERVE_TIME,
                                           len(self.no_df) - self.max_bs_number * self.max_bs_length + PRERESERVE_TIME, size=self.bs_number)
        self.bs_lengths = np.random.randint(self.min_bs_length, self.max_bs_length, size=self.bs_number)

        # Get for all the BS the current and the next prices
        self.bs_prices = []
        self.bs_future_prices = []
        for i in range(len(self.bs_lengths)):
            self.bs_prices.append(self.no_df.loc[self.bs_starts[i]:self.bs_starts[i] + self.bs_lengths[i] - 1].reset_index())
            if i == len(self.bs_lengths) - 1:  # last
                random_start = random.randint(self.bs_lengths[i], len(self.no_df))
                self.bs_future_prices.append(self.no_df.loc[random_start - self.bs_lengths[i]:random_start - 1].reset_index())
            else:
                self.bs_future_prices.append(self.no_df.loc[self.bs_starts[i + 1] - self.bs_lengths[i]:self.bs_starts[i + 1] - 1].reset_index())

        # Get the index at the beginning at t0
        timesteps_from_start = np.cumsum(self.bs_lengths) - self.bs_lengths
        prereserved_indices = [bs_start - timestep_from_start - PRERESERVE_TIME
                               for bs_start, timestep_from_start in zip(self.bs_starts, timesteps_from_start)]

        # Prereserved prices
        prereserved_prices = np.min(self.no_df.loc[prereserved_indices][['no1', 'no2', 'no3']], axis=1).to_list()
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

        return self._current_observation()

    def _current_observation(self):
        # get the current available prices
        self.current_prices = self.bs_prices[self.current_bs].loc[self.current_bs_step][['no1', 'no2', 'no3']].to_list()
        self.future_prices = self.bs_future_prices[self.current_bs].loc[self.current_bs_step][['no1', 'no2', 'no3']].to_list()
        under_or_over = self.exunov_bookings[self.current_bs]

        return [self.last_price] + self.current_prices + self.future_prices + [under_or_over, self.timestep_left_from_bs / MAX_BS_LENGTH]

    def step(self, action):
        cost = self._take_action(action)
        reward = 1 - cost

        # step to next
        self.current_bs_step += 1
        self.timestep_left_from_bs -= 1

        # If we arrived to a next bs reset everything to the next bs
        if self.timestep_left_from_bs == 0:
            # If it was the last bs`s last step we terminate the episode, only until the last-1 because we need one bs ahead for future bs price, maybe later new idea
            if self.current_bs == self.bs_number - 1:
                self.current_bs_step -= 1
                return self._current_observation(), reward, True, {}

            self.current_bs += 1
            self.current_bs_step = 0
            self.last_price = self.prereserved_prices[self.current_bs][0]
            self.timestep_left_from_bs = self.bs_lengths[self.current_bs]

        obs = self._current_observation()

        return obs, reward, False, {}

    def _take_action(self, action):
        cost = self.last_price

        if action == 0:  # do nothing
            pass

        elif action == 1:  # update reservation
            self.last_price = self.current_prices[0]
            cost += CANCELATION_FEE
        elif action == 2:  # update reservation
            self.last_price = self.current_prices[1]
            cost += CANCELATION_FEE
        elif action == 3:  # update reservation
            self.last_price = self.current_prices[2]
            cost += CANCELATION_FEE

        elif action in [4, 5, 6]:  # update reservation to first no
            if self.exunov_bookings[self.current_bs] == 0:  # exact booking, but we would like to solve the underbooking scenario
                cost += CONSTRAINT_VIOLATION_COST
            else:
                cost += CANCELATION_FEE
                if self.exunov_bookings[self.current_bs] == -1:  # underbooking
                    if action == 4:
                        new_prereserve_price = self.current_prices[0]
                    elif action == 5:
                        new_prereserve_price = self.current_prices[1]
                    elif action == 6:
                        new_prereserve_price = self.current_prices[2]

                    self.timestep_left_from_bs += self.exunov_times[self.current_bs]
                    self.bs_lengths[self.current_bs] += self.exunov_times[self.current_bs]
                    self.bs_lengths[self.current_bs + 1] -= self.exunov_times[self.current_bs]

                    # set the prices to the underbooking
                    self.bs_prices[self.current_bs] = self.no_df.loc[self.bs_prices[self.current_bs]['index'].iloc[0]:
                                                                     self.bs_prices[self.current_bs]['index'].iloc[-1] + self.exunov_times[self.current_bs]].reset_index()
                    self.bs_prices[self.current_bs + 1] = self.no_df.loc[self.bs_prices[self.current_bs + 1]['index'].iloc[0] + self.exunov_times[self.current_bs]:
                                                                         self.bs_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

                    self.bs_future_prices[self.current_bs] = self.no_df.loc[self.bs_future_prices[self.current_bs]['index'].iloc[0]:
                                                                            self.bs_future_prices[self.current_bs]['index'].iloc[-1] + self.exunov_times[self.current_bs]].reset_index()
                    self.bs_future_prices[self.current_bs + 1] = self.no_df.loc[self.bs_future_prices[self.current_bs + 1]['index'].iloc[0] + self.exunov_times[self.current_bs]:
                                                                                self.bs_future_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

                    self.prereserved_prices[self.current_bs] = self.prereserved_prices[self.current_bs] + [new_prereserve_price] * self.exunov_times[self.current_bs]
                    self.prereserved_prices[self.current_bs + 1] = self.prereserved_prices[self.current_bs + 1][self.exunov_times[self.current_bs]:]

                elif self.exunov_bookings[self.current_bs] == 1:  # overbooking
                    if action == 4:
                        new_prereserve_price = self.future_prices[0]
                    elif action == 5:
                        new_prereserve_price = self.future_prices[1]
                    elif action == 6:
                        new_prereserve_price = self.future_prices[2]

                    self.timestep_left_from_bs -= self.exunov_times[self.current_bs]
                    self.bs_lengths[self.current_bs] -= self.exunov_times[self.current_bs]
                    self.bs_lengths[self.current_bs + 1] += self.exunov_times[self.current_bs]

                    # set the prices to the over
                    self.bs_prices[self.current_bs] = self.no_df.loc[self.bs_prices[self.current_bs]['index'].iloc[0]:
                                                                     self.bs_prices[self.current_bs]['index'].iloc[-1] - self.exunov_times[self.current_bs]].reset_index()
                    self.bs_prices[self.current_bs + 1] = self.no_df.loc[self.bs_prices[self.current_bs + 1]['index'].iloc[0] - self.exunov_times[self.current_bs]:
                                                                         self.bs_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

                    self.bs_future_prices[self.current_bs] = self.no_df.loc[self.bs_future_prices[self.current_bs]['index'].iloc[0]:
                                                                            self.bs_future_prices[self.current_bs]['index'].iloc[-1] - self.exunov_times[self.current_bs]].reset_index()
                    self.bs_future_prices[self.current_bs + 1] = self.no_df.loc[self.bs_future_prices[self.current_bs + 1]['index'].iloc[0] - self.exunov_times[self.current_bs]:
                                                                                self.bs_future_prices[self.current_bs + 1]['index'].iloc[-1]].reset_index()

                    self.prereserved_prices[self.current_bs] = self.prereserved_prices[self.current_bs][:-self.exunov_times[self.current_bs]]
                    self.prereserved_prices[self.current_bs + 1] = [new_prereserve_price] * self.exunov_times[self.current_bs] + self.prereserved_prices[self.current_bs + 1]

                    # todo at the last BS no cancelation fee
                self.exunov_bookings[self.current_bs] = 0  # the under or overbooking is solved, so there is not under or overbooking

        return cost

    def _calculate_reward(self, cost: float):
        ...

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...


if __name__ == '__main__':
    no_df = load_and_preproces_datasets()
    no_df_train, no_df_val = train_val_split(no_df)
    env = BREnv(no_df_train)
    env.reset()
    env.step(4)
    env.step(1)
    env.step(1)

    asd = 0
