import math
import operator
import json
from network import Network
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from joblib import dump, load
from log_reg_network import LogReg
import gym
import env.env
from dqn import DQNAgent


# BetNet = Network()
# BetNet.load_weights("weights/Adadelta/test9_400/weights-improvement-400-0.48.hdf5") #Most Stable weights

print("LOADING BETTING POLICY...")
ENV_NAME = "BetEnv-v0"

env = gym.make(ENV_NAME)
q_table = None
with open('q_table.json') as f:
    q_table = json.load(f)

new_q_table = defaultdict( lambda: defaultdict(float))
for key in q_table.keys():
    newInner = {}
    for key2 in q_table[key]:
        newInner[eval(key2)] = q_table[key][key2]
    new_q_table[eval(key)] = newInner





###############################################################################
#
#LOAD MODEL
#
print("LOADING PREDICTION MODEL...")

#MODEL_TYPE = "NN"

# if len(sys.argv) > 1:
#     MODEL_TYPE = sys.argv[1]

#curr_model = None
def load_model(MODEL_TYPE):
    curr_model = None
    if MODEL_TYPE == "SVM":
        print("LOADING SVM...")
        curr_model = load("svm.joblib")
    elif MODEL_TYPE == "LR":
        print("LOADING LR...")
        lr = LogReg(74)#(env.matches.shape[1])
        lr.load_weights("weights/weights-improvement-100-0.31.hdf5")
        curr_model = lr
    elif MODEL_TYPE == "DT":
        print("LOADING DT...")
        curr_model = load("dt.joblib")
    elif MODEL_TYPE == "GB":
        print("LOADING GB...")
        curr_model = load("gb.joblib")
    elif MODEL_TYPE == "RF":
        print("LOADING RF...")
        curr_model = load("rfc.joblib")
    elif MODEL_TYPE == "NB":
        print("LOADING NB...")
        curr_model = load("nb.joblib")
    elif MODEL_TYPE == "AB":
        print("LOADING AB...")
        curr_model = load("ab.joblib")
    elif MODEL_TYPE == "DQN":
        print("LOADING DQN...")
        BetNet = DQNAgent(75)
        BetNet.load("weights/betnet-weights-dqn.h5")
        curr_model = BetNet
    else:
        print("LOADING NN...")
        BetNet = Network(74)#(env.matches.shape[1])
        BetNet.load_weights('weights/Adadelta/test9_400_Best/weights-improvement-400-0.48.hdf5')#PCA("weights/Adadelta/test13_100iter_reglast2/weights-improvement-100-0.52.hdf5")  # Most recent weights
        curr_model = BetNet
    return curr_model
###############################################################################

#GETS THE PREDICTION VEC GIVEN MODEL
def generatePrediction(mt, curr_model, to_process, cash):
    prediction = None
    if mt == "SVM" or mt == "DT" or mt == "GB" or mt == "NB" or mt == "RF" or mt=="AB":
        temp_pred = curr_model.predict(np.asarray(to_process))
        hardmax = np.zeros((1,3))
        hardmax[0][temp_pred[0]] = 1
        prediction = hardmax[0]
    elif mt == "LR":
        net_pred = curr_model.model.predict(np.asarray(to_process))
        idx = np.argmax(net_pred)
        hardmax = np.zeros_like(net_pred[0])
        hardmax[idx] = 1
        prediction = hardmax
    elif mt == "DQN":
        temp_pred = int(round(curr_model.getIntermediate([to_process])[0][0][0]*cash))
        full_pred = curr_model.modelReward.predict(np.asarray(to_process))
        print(full_pred)
        max_index = np.argmax(full_pred)
        hardmax = np.zeros((1, 3))
        hardmax[0][max_index] = 1
        prediction = (temp_pred, hardmax[0])
    else:
        net_pred = curr_model.model.predict(np.asarray(to_process))
        idx = np.argmax(net_pred)
        hardmax = np.zeros_like(net_pred[0])
        hardmax[idx] = 1
        prediction = hardmax
    #print(prediction)
    return prediction

home_odds_labels = ['B365H', 'BWH', 'IWH', 'LBH', 'PSH', 'WHH', 'SJH', 'VCH', 'GBH', 'BSH']
draw_odds_labels = ['B365D', 'BWD', 'IWD', 'LBD', 'PSD', 'WHD', 'SJD', 'VCD', 'GBD', 'BSD']
away_odds_labels = ['B365A', 'BWA', 'IWA', 'LBA', 'PSA', 'WHA', 'SJA', 'VCA', 'GBA', 'BSA']
all_odds_labels = [home_odds_labels, draw_odds_labels, away_odds_labels]



odd_source = pd.read_csv("data/recent_seasons_filled_unnormalized.csv")
#print(new_q_table)
print("LOADING DATA...")
x = pd.read_csv('data/recent_seasons_filled_unnormalized.csv')
#make odds list
home_odds = odd_source[home_odds_labels]
draw_odds = odd_source[draw_odds_labels]
away_odds = odd_source[away_odds_labels]
all_odds = [home_odds, draw_odds, away_odds]
top_odds_per_match = []
for i in range(0, x.shape[0]):
    match = []
    for j in range(0, 3):
        odds = all_odds[j].loc[i]
        #print(odds)
        max_odd = np.amax(odds)
        #print(max_odd)
        match.append(max_odd)
    top_odds_per_match.append(match)
#print(top_odds_per_match, len(top_odds_per_match))

x = x.drop([0], axis=0)
#x = x.drop(["Unnamed: 0"], axis=1)
y = pd.read_csv('olddata/labels_recent_seasons.csv')
y = y.drop([0], axis=0)
y = y.drop(["Unnamed: 0"], axis=1)
#print( x, y)
x, y =  x.as_matrix(), y.as_matrix()
x, y = x[320:509], y[320:]
print(x.shape, y.shape)
print("BETTING...")



def test(MODEL_TYPE):
    cash = 50
    curr_model = load_model(MODEL_TYPE)
    results = []

    for i in range(0, 50):
        ex = x[i]
        if MODEL_TYPE == "DQN":
            tempEx = ex.tolist()
            tempEx.append(cash)
            ex = np.asarray(tempEx)

        pred = generatePrediction(MODEL_TYPE, curr_model, [ex], cash)
        print(pred)
        action = None
        if MODEL_TYPE == "DQN":
            action, pred = (pred[0], np.argmax(pred[1])), pred[1]

            print(action, pred)

        if MODEL_TYPE != "DQN":
            print("wew")
            state = (tuple(pred.tolist()), cash)
            actions = new_q_table[state]
            action = None
            if len(actions.items()) > 0:
                print("set action")
                action = max(actions.items(), key=operator.itemgetter(1))[0]
            else:

                action = ( min(cash // 10, 10000), np.argmax(pred))



        odds = top_odds_per_match[i][np.argmax(y[i])]
        print(cash, action, odds, top_odds_per_match[i])
        if action[1] == np.argmax(y[i]):
            cash += (round(abs(action[0])*odds - abs(action[0])))
        else:
            cash -= abs(action[0])

        results.append(cash)
        if cash <= 0:
            print("YOU LOST ALL YOUR MONEY!!! :(")

            break




    print(cash)
    return results

models_to_test = ["DQN", "NN", "LR", "SVM", "DT"]
results = {}
for i in range(0, len(models_to_test)):
    results[models_to_test[i]] = test(models_to_test[i])

print(results)
legend = []
for key, value in results.items():
    plt.plot(value)
    legend.append(key)

plt.legend(legend, loc='upper right')
plt.ylabel('Cash')
plt.xlabel('Chronological 17-18 EPL Matches')
plt.title('Automated Betting results')
plt.show()