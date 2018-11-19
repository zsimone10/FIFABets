from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import sys

if (len(sys.argv) < 2):
    print("Please enter a valid model filename. ")



print("loading Model ")


filename = sys.argv[1]

print("Loading Dataset...")


