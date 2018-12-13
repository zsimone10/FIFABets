from gym import Env, logger
from gym import spaces
from gym.utils import colorize, seeding
from sklearn.utils import shuffle
import numpy as np
from six import StringIO
import sys
import math
import pandas as pd

class DeepBetEnv(Env):

    def __init__(self):
        self.matches, self.results = self.getSeason(" ")#TODO filename szn

        #print(self.matches)
        self.match_index = None
        self.match = None
        self.match_winner = None


        self.cash = 50

        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))
        self.observation_space = spaces.Discrete(4) # win money or lose money or make no money

        self.observation = 0

    def step(self, action):
        print("ACTION", action, self.cash)
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


        #Determine rewards, update cash
        reward = 0
        done = False
        lastCash = self.cash
        if self.observation == 2:
            self.cash += abs(action[0])
        elif self.observation == 3:
            self.cash -= abs(action[0])

        #reward = self.cash - lastCash
        if self.cash - lastCash == 0:
            reward = -5
        elif self.cash - lastCash < 0:
            reward = -10
        else:
            reward = 10

        if self.cash >= 100 or  self.match_index == self.matches.shape[0]-1: #we are done if made 100 bucks or lost all money or the season is over
            done = True
            reward = self.cash
        if self.cash <= 0:
            done = True
            reward = -10000


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
        state_to_return = self.match.tolist()
        #print(state_to_return)
        state_to_return.append(self.cash)
        #print(state_to_return)
        return np.asarray([np.asarray(state_to_return)]), reward, done, {"cash": self.cash, "Last bet on ": last_bet_on}


    def reset(self):
        self.matches, self.results = shuffle(self.matches, self.results)
        self.match_index = 0
        self.match = self.matches[self.match_index]
        self.match_winner = self.getMatchWinner()

        self.cash = 50
        self.observation = 0
        self.action_space = spaces.Tuple((spaces.Discrete(self.cash + 1), spaces.Discrete(3)))

        state_to_return = self.match.tolist()
        # print(state_to_return)
        state_to_return.append(self.cash)
        # print(state_to_return)
        return np.asarray([np.asarray(state_to_return)])

    def getSeason(self, filename):
        x = pd.read_csv('olddata/most_seasons_unnormalized.csv')
        x = x.drop([0], axis=0)
        x = x.drop(["Unnamed: 0"], axis=1)
        y = pd.read_csv('olddata/labels_most_seasons.csv')
        y = y.drop([0], axis=0)
        y = y.drop(["Unnamed: 0"], axis=1)
        #print( x, y)
        x = x.as_matrix()
        #x = x[0:319]
        y = y.as_matrix()
        #y = y[0:319]
        return x, y


    def getMatchWinner(self):
        winner_index, = np.where(self.results[self.match_index] == 1)
        return winner_index
