import gym
import operator
import math
import json
import sys
from keras.models import Sequential
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np

import sys
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from log_reg_network import LogReg
from joblib import dump, load
from network import Network
import env.env
from collections import defaultdict
from gym import spaces
from datetime import datetime
##############################################################################
#
#LOAD ENVIRONMENT
#
ENV_NAME = "BetEnv-v0"

env = gym.make(ENV_NAME)
np.random.seed(420)
env.seed(420)
nb_actions = 201 #env.action_space.n #should be 4

###############################################################################
#
#LOAD MODEL
#

MODEL_TYPE = "NN"

if len(sys.argv) > 1:
    MODEL_TYPE = sys.argv[1]

curr_model = None
if MODEL_TYPE == "SVM":
    print("LOADING SVM...")
    curr_model = load("svm.joblib")
elif MODEL_TYPE == "LR":
    print("LOADING LR...")
    lr = LogReg(env.matches.shape[1])
    lr.load_weights("weights/lr/batch_run_1/99weights-improvement-100-0.49.hdf5")
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
else:
    print("LOADING NN...")
    BetNet = Network(env.matches.shape[1])
    BetNet.load_weights("weights/Adadelta/test13_100iter_reglast2/weights-improvement-100-0.52.hdf5")  # Most recent weights
    curr_model = BetNet

###############################################################################

#GETS THE PREDICTION VEC GIVEN MODEL
def generatePrediction(mt, curr_model, to_process):
    prediction = None
    if mt == "SVM" or mt == "DT" or mt == "GB" or mt == "NB" or mt == "RF" or mt=="AB":
        temp_pred = curr_model.predict(np.asarray([np.transpose(to_process[0])]))
        hardmax = np.zeros((1,3))
        hardmax[0][temp_pred[0]] = 1
        prediction = tuple(hardmax[0].tolist())
    elif mt == "LR":
        net_pred = curr_model.model.predict(np.asarray([np.transpose(to_process[0])]))
        idx = np.argmax(net_pred)
        hardmax = np.zeros_like(net_pred[0])
        hardmax[idx] = 1
        prediction = tuple(hardmax.tolist())
    elif mt == "DT":
        temp_pred = curr_model.predict(np.asarray([np.transpose(to_process[0])]))
        hardmax = np.zeros((1, 3))
        hardmax[0][temp_pred[0]] = 1
        prediction = tuple(hardmax[0].tolist())
    else:
        net_pred = curr_model.model.predict(np.asarray([np.transpose(to_process[0])]))
        idx = np.argmax(net_pred)
        hardmax = np.zeros_like(net_pred[0])
        hardmax[idx] = 1
        prediction = tuple(hardmax.tolist())
    #print(prediction)
    return prediction

PREDICT_BET = True
if len(sys.argv) > 2:
    print("PREDICT_BET: ", sys.argv[2])
    PREDICT_BET = bool(sys.argv[2])


q_table = defaultdict( lambda: defaultdict(float))

import random
# from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.7
epsilon = 0.4

# For plotting metrics
all_epochs = []
all_penalties = []
starttime = datetime.now()
for i in range(1, 30001):
    state_to_process = env.reset()
    #print("RESET", state_to_process, env.action_space.spaces[1].n)
    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:

        def makeState(to_process):
            prediction = generatePrediction(MODEL_TYPE, curr_model, to_process)

            return (prediction, to_process[1])

        state = makeState(state_to_process)
        #print(state)
        def getMaxAction(state_for_action):
            newAct = None
            #print(state_for_action, q_table[state_for_action].items())
            for act in max(q_table[state_for_action].items(), key=operator.itemgetter(1)):
                #print("max", act, env.cash)
                #print(act)
                if newAct or type(act) != tuple:
                    break
                if act[0] <= env.cash and act[0] > 0:
                    newAct = act

            if not newAct:
                action = env.action_space.sample()
                if PREDICT_BET:
                    #print(action)
                    temp_act = [action[0]]
                    temp_act.append(np.argmax(np.asarray(state_for_action[0])))  # Explore action space
                    newAct = tuple(temp_act)
                    #print(action, "\n\n")

            return newAct

        def makeAction(state_for_action, rand=True):
            action = None
            #print(q_table[state_for_action])
            if rand:
                if random.uniform(0, 1) < epsilon or not q_table[state_for_action]:

                    action = env.action_space.sample()
                    if PREDICT_BET:
                        #print(action)
                        temp_act = [action[0]]
                        temp_act.append(np.argmax(np.asarray(state_for_action[0])))#Explore action space
                        action = tuple(temp_act)
                        #print(action, "\n\n")
                        #print(action)
                else:
                   action = getMaxAction(state_for_action)
            else:
                if not q_table[state_for_action]:
                    action = env.action_space.sample()  # Explore action space
                else:
                    action = getMaxAction(state_for_action)
           # print(action)

            return action

        action = makeAction(state)

        next_state_to_process, reward, done, info = env.step(action) #STEP
        #print(next_state_to_process, reward, done, info)
        if done: #DONE
            #print("DONE")
            break

        old_value = q_table[state][action] #TODO
        next_state = makeState(next_state_to_process)
        next_action = makeAction(next_state, False)
        next_max = q_table[next_state][next_action] #TODO

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state][action] = new_value

        # if reward == -10:
        #     penalties += 1

        state_to_process = next_state_to_process
        epochs += 1

    if i % 100 == 0:
        #clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
print("Time Taken: ", datetime.now()-starttime)

print(q_table)
new_q_table = {}
for key in q_table.keys():
    newInner = {}
    for key2 in q_table[key]:
        newInner[str(key2)] = q_table[key][key2]
    new_q_table[str(key)] = newInner


#print(q_table)
#json_str = json.dumps(dict(q_table))
with open("q_table.json", 'w') as f:
    json.dump(dict(new_q_table), f, indent=4, separators=(',', ': '))

















""" #THIS IS THE DQN MODEL TO FIGURE OUT LATER
memory = SequentialMemory(limit=50000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=BetNet.model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

adadelta = optimizers.Adadelta()
dqn.compile(optimizer=adadelta, metrics=['accuracy'])

dqn.fit(env, nb_steps=50000, visualize=True, verbose=2)

dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)"""