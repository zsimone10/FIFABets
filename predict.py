from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, label_ranking_average_precision_score
import itertools
from network import Network
from log_reg_network import LogReg
seed = np.random.seed(7)
if (len(sys.argv) < 2):
    print("Please enter a valid model filename. ")



print("loading Model ")


filename = sys.argv[1]
MODEL = sys.argv[2]

#########################################
# Load Data
print("LOADING DATASETS")
def load_data(path):
    data = pd.read_csv(path, header=None)
    #data = data.drop([0], axis=1)
    data = data.drop([0], axis=0)
    return data.as_matrix()
x = load_data('data/recent_seasons_PCA_99_pct_44_components.csv')
print(x)
y = load_data('data/labels_recent_seasons.csv')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=seed)
#CLASSES DEFINE
le = LabelEncoder()
le.fit(["Away Win", "Draw", "Home Win"])
#########################################
#Build Network
#
#DETAILS:
#
#
model = None
if MODEL == "NN":
    model = Network(x.shape[1])
else:
    model = LogReg(x.shape[1])

model.load_weights(filename)
#########################################
model.eval(x, y, le)
