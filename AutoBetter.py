import math
import operator
import json
from network import Network
from collections import defaultdict
import numpy as np
import pandas as pd

print("LOADING PREDICTION MODEL...")
BetNet = Network()
BetNet.load_weights("weights/weights-improvement-5000-0.45.hdf5") #Most recent weights

print("LOADING BETTING POLICY...")

q_table = None
with open('q_table.json') as f:
    q_table = json.load(f)

new_q_table = defaultdict( lambda: defaultdict(float))
for key in q_table.keys():
    newInner = {}
    for key2 in q_table[key]:
        newInner[float(key2)] = q_table[key][key2]
    new_q_table[eval(key)] = newInner
#print(new_q_table)
print("LOADING DATA...")
x = pd.read_csv('cleaned_data_1718.csv')
x = x.drop([0], axis=0)
x = x.drop(["Unnamed: 0"], axis=1)
y = pd.read_csv('labels_1718.csv')
y = y.drop([0], axis=0)
y = y.drop(["Unnamed: 0"], axis=1)
# print( x, y)
x, y =  x.as_matrix(), y.as_matrix()

print("BETTING...")
cash = 50
for i in range(0, x.shape[0]):
    ex = x[i]
    pred = BetNet.model.predict(np.asarray([ex]))
    idx = np.argmax(pred)
    hardmax = np.zeros_like(pred[0])
    hardmax[idx] = 1
    print(hardmax, y[i])
    correct = False
    if hardmax.tolist() == y[i].tolist():
        #print("good pred")
        correct = True
    state = (tuple(hardmax.tolist()), cash)
    #print(state)
    actions = new_q_table[state]
    #print(dict(actions))
    action = max(actions.items(), key=operator.itemgetter(1))[0]
    print(cash, action)
    if correct:
        cash += abs(action)
    else:
        cash -= abs(action)
    #cash += action

    if cash <= 0:
        print("YOU LOST ALL YOUR MONEY!!! :(")
        break

print(cash)