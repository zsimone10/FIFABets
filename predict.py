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

seed = np.random.seed(7)
if (len(sys.argv) < 2):
    print("Please enter a valid model filename. ")



print("loading Model ")


filename = sys.argv[1]


#########################################
# Load Data
print("LOADING DATASETS")
def load_data(path):
    data = pd.read_csv(path, header=None)
    data = data.drop([0], axis=1)
    data = data.drop([0], axis=0)
    return data.as_matrix()
x = load_data('cleaned_data.csv')
print(x)
y = load_data('labels.csv')
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=seed)
#CLASSES DEFINE
le = LabelEncoder()
le.fit(["Home Win", "Draw", "Away Win"])
#########################################
#Build Network
#
#DETAILS:
#
#
model = Network()
    model.load_weights(filename)
#########################################
model.predict(x_test, y_test, le)
