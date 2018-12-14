from gym import Env, logger
from gym import spaces
from gym.utils import colorize, seeding
import numpy as np
from six import StringIO
import sys
import math
import pandas as pd
from network import Network

class FA_Env(Env):

    def __init__(self):
        self.matches, self.results, self.odds = self.getSeason('recent_seasons_filled_unnormalized.csv', 'recent_seasons_PCA_99_pct_44_components.csv')

        #print(self.matches)
        self.match_index = None
        self.match = None
        self.match_winner = None


        self.cash = 50

        self.action_space = spaces.Tuple((spaces.Discrete(math.floor(self.cash)), spaces.Discrete(3)))

        self.observation = 0

        BetNet = Network(self.matches.shape[1])
        BetNet.load_weights("weights-improvement-100-0.52.hdf5")

        self.predictions = np.zeros(self.results.shape)
        for r in range(self.matches.shape[0]):
            match = self.matches[r]
            self.predictions[r] = BetNet.model.predict(np.array([match]))[0]


    def step(self, action):
        print(self.action_space)
        print(action)
        assert self.action_space.contains(action)
        bet_amount, bet_team = action

        #Determine rewards, update cash
        reward = 0
        done = False
        lastCash = self.cash
        curr_odds = self.odds[self.match_index][action[1]]
        if bet_team == self.match_winner:
            self.cash += bet_amount * (curr_odds - 1)
        else:
            self.cash -= bet_amount

        reward = self.cash - lastCash
        if self.cash <= 1 or self.match_index == self.matches.shape[0]-1:
            done = True
            reward = self.cash


        #Update the State
        #set match index += 1
        #set match = self.matches[self.match_index]
        #update match_winner
        if not done:
            self.match_index += 1
            self.match = self.matches[self.match_index]
            self.match_odds = self.odds[self.match_index]
            self.match_predictions = self.predictions[self.match_index]
            self.match_winner = self.getMatchWinner()
            self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        return (self.match, self.match_predictions, self.match_odds, self.cash), reward, done, {"cash": self.cash}


    def reset(self):
        self.match_index = 0
        self.match = self.matches[self.match_index]
        self.match_odds = self.odds[self.match_index]
        self.match_predictions = self.predictions[self.match_index]
        self.match_winner = self.getMatchWinner()

        self.cash = 50
        self.observation = 0
        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        return (self.match, self.match_predictions, self.match_odds, self.cash)

    def getSeason(self, match_source, NN_train_source):

        top_odds_per_match = pd.read_csv("data/recent_seasons_max_odds.csv")
        print("LOADING DATA...")
        x = pd.read_csv('data/recent_seasons_PCA_99_pct_44_components.csv')


        y = pd.read_csv('data/labels_recent_seasons.csv')
        #print( x, y)
        x = x.as_matrix()
        y = y.as_matrix()
        top_odds_per_match = top_odds_per_match.as_matrix()
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
