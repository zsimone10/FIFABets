from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import sys
from network import Network


np.random.seed(7)

numEpochs = 500
weights = None
if len(sys.argv) > 1:
     numEpochs = sys.argv[1]
if len(sys.argv) > 2:
    weights = sys.argv[2]

#########################################
#TODO: Write dataset loading
# Generate dummy data
print("LOADING DATASETS")
def load_data(path):
    data = pd.read_csv(path, header=None)
    data = data.drop([0], axis=1)
    data = data.drop([0], axis=0)
    return data.as_matrix()
x = load_data('cleaned_data.csv')
print(x)
y = load_data('labels.csv')

#########################################
#Build Network
#
#DETAILS:
#
#
model = Network()
if weights:
    model.load_weights(weights)
#########################################
#Train
#
#
history = model.train(x, y, numEpochs)

#########################################
#Run Metrics
#score = model.evaluate(x_test, y_test, batch_size=128)
# Plot training & validation accuracy values

#########################################
#Save Weights
#


