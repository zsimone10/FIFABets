import math
import operator
import json
from network import Network
from collections import defaultdict
import numpy as np
import pandas as pd
import sys
from joblib import dump, load


# BetNet = Network()
# BetNet.load_weights("weights/Adadelta/test9_400/weights-improvement-400-0.48.hdf5") #Most recent weights

print("LOADING BETTING POLICY...")

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

MODEL_TYPE = "NN"

if len(sys.argv) > 1:
    MODEL_TYPE = sys.argv[1]

curr_model = None
if MODEL_TYPE == "SVM":
    print("LOADING SVM...")
    curr_model = load("SVM.joblib")
elif MODEL_TYPE == "LR":
    print("LOADING LR...")

elif MODEL_TYPE == "DT":
    print("LOADING DT...")
    curr_model = load("DT.joblib")
else:
    print("LOADING NN...")
    BetNet = Network()
    BetNet.load_weights("weights/Adadelta/test9_400/weights-improvement-400-0.48.hdf5")  # Most recent weights
    curr_model = BetNet
###############################################################################

#GETS THE PREDICTION VEC GIVEN MODEL
def generatePrediction(mt, curr_model, to_process):
    prediction = None
    if mt == "SVM":
        temp_pred = curr_model.predict(np.asarray(to_process))
        hardmax = np.zeros((1,3))
        hardmax[0][temp_pred[0]] = 1
        prediction = hardmax[0]
    elif mt == "LR":
        pass
    elif mt == "DT":
        temp_pred = curr_model.predict(np.asarray(to_process))
        hardmax = np.zeros((1, 3))
        hardmax[0][temp_pred[0]] = 1
        prediction = hardmax[0]
    else:
        net_pred = curr_model.model.predict(np.asarray(to_process))
        idx = np.argmax(net_pred)
        hardmax = np.zeros_like(net_pred[0])
        hardmax[idx] = 1
        prediction = hardmax
    #print(prediction)
    return prediction





#print(new_q_table)
print("LOADING DATA...")
x = pd.read_csv('data/recent_seasons_unnormalized.csv')
x = x.drop([0], axis=0)
x = x.drop(["Unnamed: 0"], axis=1)
y = pd.read_csv('data/labels_recent_seasons.csv')
y = y.drop([0], axis=0)
y = y.drop(["Unnamed: 0"], axis=1)
#print( x, y)
x, y =  x.as_matrix(), y.as_matrix()
x, y = x[320:], y[320:]
print("BETTING...")




cash = 50



for i in range(0, x.shape[0]):
    ex = x[i]
    pred = generatePrediction(MODEL_TYPE, curr_model, [ex])
    correct = False
    if pred.tolist() == y[i].tolist():
        #print("good pred")
        correct = True
    print(pred, y[i])
    state = (tuple(pred.tolist()), cash)
    #print(state)
    actions = new_q_table[state]
    #print(actions)
    #print(dict(actions))
    action = None
    if len(actions.items()) > 0:
        print("set action")
        action = max(actions.items(), key=operator.itemgetter(1))[0]
    else:
        action = (1, np.argmax(pred))
    print(cash, action)
    #print(action[1], np.argmax(y[i]))
    if action[1] == np.argmax(y[i]):
        cash += abs(action[0])
    else:
        cash -= abs(action[0])
    #cash += action

    if cash <= 0:
        print("YOU LOST ALL YOUR MONEY!!! :(")
        break

print(cash)