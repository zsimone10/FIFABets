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
        self.matches, self.results, self.odds = self.getSeason(" ")#TODO filename szn

        #print(self.matches)
        self.match_index = None
        self.match = None
        self.match_winner = None


        self.cash = 50

        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))
        self.observation_space = spaces.Discrete(4) # win money or lose money or make no money

        self.observation = 0

    def step(self, action):
        #print("ACTION", action, self.cash)
        assert self.action_space.contains(action)
        #print(action)
        # action[0] = math.ceil(action[0])
        #check to see if our prediction is correct on the current match
        last_bet_on = None
        if action[0] == 0: #no change in money
            self.observation = 1
        elif (action[1] == self.match_winner): #we make money
            self.observation = 2
        else: #we lose money
            self.observation = 3


        # # determine who the last team we bet on is
        # #print(action[0], self.cash)
        # if action[0] < 0:
        #     last_bet_on = "Away"
        # elif action[0] > 0:
        #     last_bet_on = "Home"
        # else:
        #     last_bet_on == "no one"
        #

        #Determine rewards, update cash
        reward = 0
        done = False
        lastCash = self.cash
        curr_odds = self.odds[self.match_index][action[1]]
        if self.observation == 2:
            self.cash += (round(abs(action[0])*curr_odds))-abs(action[0])
        elif self.observation == 3:
            self.cash -= abs(action[0])

        reward = self.cash - lastCash
        if self.cash >= 500 or self.cash <= 0 or self.match_index == self.matches.shape[0]-1: #we are done if made 100 bucks or lost all money or the season is over
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
            self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        #print((self.match, self.cash), reward, done)
        return (self.match, self.cash), reward, done, {"cash": self.cash, "Last bet on ": last_bet_on}


    def reset(self):
        self.match_index = 0
        self.match = self.matches[self.match_index]
        self.match_winner = self.getMatchWinner()

        self.cash = 50
        self.observation = 0
        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        return (self.match, self.cash)

    def getSeason(self, filename):
        home_odds_labels = ['B365H', 'BWH', 'IWH', 'LBH', 'PSH', 'WHH', 'SJH', 'VCH', 'GBH', 'BSH']
        draw_odds_labels = ['B365D', 'BWD', 'IWD', 'LBD', 'PSD', 'WHD', 'SJD', 'VCD', 'GBD', 'BSD']
        away_odds_labels = ['B365A', 'BWA', 'IWA', 'LBA', 'PSA', 'WHA', 'SJA', 'VCA', 'GBA', 'BSA']
        all_odds_labels = [home_odds_labels, draw_odds_labels, away_odds_labels]
        odd_source = pd.read_csv("data/recent_seasons_filled_unnormalized.csv")
        # print(new_q_table)
        print("LOADING DATA...")
        x = pd.read_csv('data/recent_seasons_filled_unnormalized.csv')
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
