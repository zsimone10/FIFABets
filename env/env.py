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
        #     last_bet_on == "Away"
        # elif action[0] > 0:
        #     last_bet_on == "Home"
        # else:
        #     last_bet_on == "no one"
        #

        #Determine rewards, update cash
        reward = 0
        done = False
        lastCash = self.cash
        if self.observation == 2:
            self.cash += abs(action[0])
        elif self.observation == 3:
            self.cash -= abs(action[0])

        reward = self.cash - lastCash
        if self.cash >= 100 or self.cash <= 0 or self.match_index == self.matches.shape[0]-1: #we are done if made 100 bucks or lost all money or the season is over
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
        x = pd.read_csv('data/recent_seasons_unnormalized.csv')
        x = x.drop([0], axis=0)
        x = x.drop(["Unnamed: 0"], axis=1)
        y = pd.read_csv('data/labels_recent_seasons.csv')
        y = y.drop([0], axis=0)
        y = y.drop(["Unnamed: 0"], axis=1)
        #print( x, y)
        x = x.as_matrix()
        x = x[0:319]
        y = y.as_matrix()
        y = y[0:319]
        return x, y


    def getMatchWinner(self):
        winner_index, = np.where(self.results[self.match_index] == 1)
        return winner_index

# TODO DQN ENV
# class BetEnv1(Env):
#
#     def __init__(self):
#         self.matches, self.results = self.getSeason(" ")#TODO filename szn
#
#         #print(self.matches)
#         self.match_index = None
#         self.match = None
#         self.match_winner = None
#
#
#         self.cash = 50
#
#         self.action_space = spaces.Box(low=np.array([-self.cash]), high=np.array([self.cash]),)
#         self.observation_space = spaces.Discrete(4) # win money or lose money or make no money
#
#         self.observation = 0
#
#     def step(self, action):
#         print
#         assert self.action_space.contains(action)
#         action[0] = math.ceil(action[0])
#         #check to see if our prediction is correct on the current match
#         last_bet_on = None
#         if action[0] == 0: #no change in money
#             self.observation = 1
#         elif (action[0] < 0 and self.match_winner == 2) or (action[0] > 0 and self.match_winner == 0): #we make money
#             self.observation = 2
#         else: #we lose money
#             self.observation = 3
#
#
#         # determine who the last team we bet on is
#         #print(action[0], self.cash)
#         if action[0] < 0:
#             last_bet_on == "Away"
#         elif action[0] > 0:
#             last_bet_on == "Home"
#         else:
#             last_bet_on == "no one"
#
#
#         #Determine rewards, update cash
#         reward = 0
#         done = False
#         lastCash = self.cash
#         if self.observation == 2:
#             self.cash += abs(action[0])
#         elif self.observation == 3:
#             self.cash -= abs(action[0])
#
#         reward = self.cash - lastCash
#         if self.cash >= 100 or self.cash <= 0 or self.match_index == self.matches.shape[0]-1: #we are done if made 100 bucks or lost all money or the season is over
#             done = True
#             reward = self.cash
#
#
#         #Update the State
#         #set match index += 1
#         #set match = self.matches[self.match_index]
#         #update match_winner
#         if not done:
#             self.match_index += 1
#             self.match = self.matches[self.match_index]
#             self.match_winner = self.getMatchWinner()
#             self.action_space = spaces.Box(low=np.array([-self.cash]), high=np.array([self.cash]))
#
#         return (self.match, self.cash), reward, done, {"cash": self.cash, "Last bet on ": last_bet_on}
#
#
#     def reset(self):
#         self.match_index = 0
#         self.match = self.matches[self.match_index]
#         self.match_winner = self.getMatchWinner()
#
#         self.cash = 50
#         self.observation = 0
#         return (self.match, self.cash)
#
#     def getSeason(self, filename):
#         x = pd.read_csv('cleaned_data.csv')
#         x = x.drop([0], axis=0)
#         x = x.drop(["Unnamed: 0"], axis=1)
#         y = pd.read_csv('labels.csv')
#         y = y.drop([0], axis=0)
#         y = y.drop(["Unnamed: 0"], axis=1)
#         print( x, y)
#         return x.as_matrix(), y.as_matrix()
#
#
#     def getMatchWinner(self):
#         winner_index, = np.where(self.results[self.match_index] == 1)
#         return winner_index