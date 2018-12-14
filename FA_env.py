from gym import Env, logger
from gym import spaces
from gym.utils import colorize, seeding
import numpy as np
from six import StringIO
import sys
import math
import pandas as pd

class BetEnv(Env):

    def __init__(self):
        self.matches, self.results, self.odds = self.getSeason('recent_seasons_filled_unnormalized.csv', 'recent_seasons_PCA_99_pct_44_components.csv')

        #print(self.matches)
        self.match_index = None
        self.match = None
        self.match_winner = None


        self.cash = 50

        self.action_space = spaces.Tuple(spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        self.observation = 0

    def step(self, action):
        assert self.action_space.contains(action)
        bet_amount, bet_team = action

        #Determine rewards, update cash
        reward = 0
        done = False
        lastCash = self.cash
        curr_odds = self.odds[self.match_index][action[1]]
        if bet_team == match.winner:
            self.cash += bet_amount * (curr_odds - 1)
        else:
            self.cash -= bet_amount

        reward = self.cash - lastCash
        if self.cash <= 0 or self.match_index == self.matches.shape[0]-1:
            done = True
            reward = self.cash


        #Update the State
        #set match index += 1
        #set match = self.matches[self.match_index]
        #update match_winner
        if not done:
            self.match_index += 1
            self.match = self.matches[self.match_index]
            self.match_winner = self.getMatchWinner()
            self.action_space = spaces.Tuple(spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        return (self.match, self.cash), reward, done, {"cash": self.cash}


    def reset(self):
        self.match_index = 0
        self.match = self.matches[self.match_index]
        self.match_winner = self.getMatchWinner()

        self.cash = 50
        self.observation = 0
        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))
        odds = match.getOdds()
        win_probs = match.getWinProbs()

        return (self.match, self.cash)

    def getSeason(self, match_source, NN_train_source):
        home_odds_labels = ['B365H', 'BWH', 'IWH', 'LBH', 'PSH', 'WHH', 'SJH', 'VCH', 'GBH', 'BSH']
        draw_odds_labels = ['B365D', 'BWD', 'IWD', 'LBD', 'PSD', 'WHD', 'SJD', 'VCD', 'GBD', 'BSD']
        away_odds_labels = ['B365A', 'BWA', 'IWA', 'LBA', 'PSA', 'WHA', 'SJA', 'VCA', 'GBA', 'BSA']
        all_odds_labels = [home_odds_labels, draw_odds_labels, away_odds_labels]
        odd_source = pd.read_csv("data/recent_seasons_filled_unnormalized.csv")
        # print(new_q_table)
        print("LOADING DATA...")
        x = pd.read_csv('data/recent_seasons_PCA_99_pct_44_components.csv')
        # make odds list
        home_odds = odd_source[home_odds_labels]
        draw_odds = odd_source[draw_odds_labels]
        away_odds = odd_source[away_odds_labels]
        all_odds = [home_odds, draw_odds, away_odds]
        top_odds_per_match = []
        for i in range(0, x.shape[0]):
            match = []
            for j in range(0, 3):
                odds = all_odds[j].loc[i]
                # print(odds)
                max_odd = np.amax(odds)
                # print(max_odd)
                match.append(max_odd)
            top_odds_per_match.append(match)
        # print(top_odds_per_match, len(top_odds_per_match))

        x = x.drop([0], axis=0)
        #x = x.drop(["Unnamed: 0"], axis=1)
        y = pd.read_csv('olddata/labels_recent_seasons.csv')
        y = y.drop([0], axis=0)
        y = y.drop(["Unnamed: 0"], axis=1)
        #print( x, y)
        x = x.as_matrix()
        x = x[0:319]
        y = y.as_matrix()
        y = y[0:319]
        return x, y, top_odds_per_match


    def getMatchWinner(self):
        winner_index, = np.where(self.results[self.match_index] == 1)
        return winner_index


#######################################################################################
#
#
#
#
#
########################################################################################
# TODO DQN ENV
