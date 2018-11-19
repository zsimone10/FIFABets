from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import sys

np.random.seed(7)

class Network:
    def __init__(self):
        self.model = self.create()

    def create(self):
        print("BUILDING NETWORK...")

        model = Sequential()
        model.add(Dense(512, input_dim=78))
        model.add(Activation('sigmoid'))
        model.add(Dense(128))
        model.add(Activation('sigmoid'))
        model.add(Dense(128))
        model.add(Activation('sigmoid'))
        model.add(Dense(64))
        model.add(Activation('sigmoid'))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        plot_model(model, to_file='model.png')
        print(model.summary())
        return model

    def train(self, x, y, epochs=500, period=250):
        print("TRAINING...")
        #sgd = optimizers.SGD(lr=0.09)
        adadelta = optimizers.Adadelta()
        self.model.compile(loss='categorical_crossentropy', optimizer=adadelta,
                      metrics=['accuracy'])


        filepath = "weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False,
                                     save_weights_only=False, mode='auto', period=period)

        callbacks_list = [checkpoint]
        history = self.model.fit(x, y,
                            validation_split=0.1,  # initially .25
                            epochs=epochs,
                            batch_size=10,
                            callbacks=callbacks_list)

        return history

    def load_weights(self, weights):
        self.model.load_weights(weights)



