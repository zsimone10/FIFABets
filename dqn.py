import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, concatenate, LeakyReLU
from keras.utils import plot_model
from keras import backend as K
from keras.constraints import maxnorm, nonneg
from keras import regularizers

from keras import optimizers
import env.deepenv
from collections import deque
import random
from datetime import datetime

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy


class DQNAgent:
    def __init__(self, state_size, action_size=3):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=250)
        self.gamma = 0.95    # discount rate originally: [.95]
        self.epsilon = 0.4 # exploration rate originally [1.0]
        self.epsilon_min = 0.01
        self.epsilon_decay = 1.0 #.995 #[0.995] 1 means no decay
        self.learning_rate = 0.01 #[0.001]
        self.modelReward, self.modelPred = self._build_model(state_size)

    def _build_model(self, data_dim):
        # Neural Net for Deep-Q learning Model
        minputs = Input(shape=(data_dim, ))
        # $ Arm
        a = Dense(64, activation='sigmoid')(minputs)
        a = Dense(64, activation='sigmoid')(a)
        a = Dense(32, activation='sigmoid')(a)
        a = Dense(32, activation='sigmoid')(a)
        a = Dense(1, activation='sigmoid', W_constraint = nonneg())(a)


        #reward comilation
        r = concatenate([a, minputs])
        r = Dense(512, activation='relu')(r)
        r = Dense(128, activation='relu')(r)
        r = Dense(128, activation='relu')(r)
        r = Dense(64, activation='relu')(r)
        output = Dense(3, activation='linear')(r)
        modelReward = Model(inputs=minputs, outputs=output)
        modelPred = Model(inputs=modelReward.input, outputs=[modelReward.get_layer('dense_5').output])
        plot_model(modelReward, to_file='modelDQNfull.png')
        plot_model(modelPred, to_file='modelDQNPred.png')
        plot_model(modelReward, to_file='dqnmodel.png', show_shapes=True, show_layer_names=False)

        print(modelReward.summary())
        adadelta = optimizers.Adadelta()
        modelReward.compile(loss='mse',
                      optimizer=adadelta, metrics=['accuracy'])
        modelPred.compile(loss='mse',
                            optimizer=adadelta, metrics=['accuracy'])
        return modelReward, modelPred

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def getIntermediate(self, state):
        getOutput = K.function(inputs=[self.modelReward.input], outputs=[self.modelReward.get_layer('dense_5').output])

        return getOutput(state)

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            #print('rand', env.action_space.sample())
            return env.action_space.sample()
        print(self.getIntermediate([state])[0][0][0])
        predicted_bet = int(round(self.getIntermediate([state])[0][0][0]*env.cash))
        predicted_highest = np.argmax(self.modelReward.predict(state))

        return (predicted_bet, predicted_highest)  # returns action

    def replay(self, batch_size):
        #print("REPLAY")
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.modelReward.predict(next_state)))
            target_f = self.modelReward.predict(state)
            target_f[0][action[1]] = target
            self.modelReward.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        print("LOADING WEIGHTS...")
        self.modelReward.load_weights(name)

    def save(self, name):
        print("SAVING WEIGHTS...")
        self.modelReward.save_weights(name)

EPISODES = 201

if __name__ == "__main__":
    env = gym.make('DeepBetEnv-v0')
    state_size = 75
    action_size = 3
    agent = DQNAgent(state_size, action_size)
    #agent.load("weights/BetnetWeights/betnetweights1/betnet-weights-dqn.h5")
    done = False
    batch_size = 32
    starttime = datetime.now()
    for e in range(EPISODES):
        state = env.reset()
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}, final cash: {}"
                      .format(e, EPISODES, time, agent.epsilon, env.cash))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if time % 100 == 0:
                print(time)
        if e % 10 == 0:
            agent.save("weights/betnet-weights-dqn.h5")

    print("Training finished.\n")
    print("Time Taken: ", datetime.now() - starttime)

