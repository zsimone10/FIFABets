import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, label_ranking_average_precision_score
import itertools
from sklearn.utils import resample
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from joblib import dump, load


#most_seasons_PCA_99_pct_44_components.csv
#recent_seasons_PCA_99_pct_44_components.csv

#most_seasons_PCA_95_pct_34_components.csv
#recent_seasons_PCA_95_pct_34_components.csv

#most_seasons_PCA_90_pct_27_components.csv
#recent_seasons_PCA_90_pct_27_components.csv

#most_seasons_PCA_85_pct_21_components.csv
#recent_seasons_PCA_85_pct_21_components.csv

#most_seasons_PCA_75_pct_13_components.csv
#recent_seasons_PCA_75_pct_13_components.csv


seed = np.random.seed(7)

print("LOADING DATASETS")
def load_data(path):
    data = pd.read_csv(path)
    #data = data.drop([0], axis=1)
    #data = data.drop([0], axis=0)
    return data
    #RETURN data.as_matrix()
x = load_data('data/most_seasons_PCA_99_pct_44_components.csv')
#print(x)
y = load_data('olddata/labels_most_seasons.csv')


x1 = load_data('data/recent_seasons_PCA_99_pct_44_components.csv')
#print(x)
y1 = load_data('olddata/labels_recent_seasons.csv')

def parseData(x, y, c, resampling):
############
#RESAMPLE
    data = pd.concat([x, y], axis=1)
#data = data.drop([0], axis=1)

#data = data.drop([0], axis=0)
#print(data)

    home_wins = data[data["Home"] == 1]
    draws = data[data["Draw"] == 1]
    away_wins = data[data["Away"] == 1]
    print(len(home_wins), len(draws), len(away_wins))

    if resampling:
        draws_resampled = resample(draws,
                                    replace=True,     # sample with replacement
                                    n_samples=math.ceil(len(home_wins)*0.8),    # to half home_wins
                                    random_state=123) # reproducible results
        away_wins_resampled = resample(away_wins,
                                    replace=True,     # sample with replacement
                                    n_samples=math.ceil(len(home_wins)*0.85),    # to match majority class
                                    random_state=123) # reproducible results

        print(len(home_wins), len(draws_resampled), len(away_wins_resampled))
        resampled = pd.concat([draws_resampled, away_wins_resampled])
#print(resampled)
    else:
        resampled =  pd.concat([draws, away_wins])

    data = pd.concat([home_wins, resampled])


    y = data[["Home", "Draw", "Away"]]
    x = data.drop(["Unnamed: 0", "Home", "Draw", "Away"], axis=1)

#print(x, y)
    y = y.drop([0], axis=0)
#x = x.drop([0], axis=1)
    x = x.drop([0], axis=0)

#print(x, y)
    x, y = x.as_matrix(), y.as_matrix()

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=c, random_state=seed)

    return x_train, x_test, y_train, y_test

##############

x_train, x_test, y_train, y_test = parseData(x,y, 0.9, resampling = 1)
x_trainR, x_testR, y_trainR, y_testR = parseData(x1,y1, 0.0, resampling = 0)


############
#RESAMPLE



##############
sc = StandardScaler()
scaledX_train = sc.fit_transform(x_train)
scaledX_test = sc.transform(x_test)
scaledXR_test = sc.transform(x_testR)

changed_y_train_list = []
changed_y_test_list = []
changed_y_train_listR = []
changed_y_test_listR = []

print(len(y_train))

for i in range(len(y_train)):
    if y_train[i][0] == 1:
        changed_y_train_list.append(0)
    if y_train[i][1] == 1:
        changed_y_train_list.append(1)
    if y_train[i][2] == 1:
        changed_y_train_list.append(2)

for i in range(len(y_test)):
    if y_test[i][0] == 1:
        changed_y_test_list.append(0)
    if y_test[i][1] == 1:
        changed_y_test_list.append(1)
    if y_test[i][2] == 1:
        changed_y_test_list.append(2)

for i in range(len(y_testR)):
    if y_testR[i][0] == 1:
        changed_y_test_listR.append(0)
    if y_testR[i][1] == 1:
        changed_y_test_listR.append(1)
    if y_testR[i][2] == 1:
        changed_y_test_listR.append(2)

for i in range(len(y_trainR)):
    if y_trainR[i][0] == 1:
        changed_y_train_listR.append(0)
    if y_trainR[i][1] == 1:
        changed_y_train_listR.append(1)
    if y_trainR[i][2] == 1:
        changed_y_train_listR.append(2)



y_train1 = np.asarray(changed_y_train_list)
y_test1 = np.asarray(changed_y_test_list)
y_trainRe = np.asarray(changed_y_train_listR)
y_testRe = np.asarray(changed_y_test_listR)

#sc = StandardScaler()
#scaledX_train = sc.fit_transform(x_train)
#scaledX_test = sc.transform(x_test)



le = LabelEncoder()
le.fit(["Home Win", "Draw", "Away Win"])
print(le.classes_)

#train Decision Tree

rfc = RandomForestClassifier()

rfc.fit(scaledX_train, y_train1)

#tree.export_graphviz(rfc, out_file='tree.dot', class_names=["Home Win", "Draw", "Away Win"], leaves_parallel = True, )

y_pred_train = rfc.predict(scaledX_train)
y_pred_test = rfc.predict(scaledX_test)
y_pred_testR = rfc.predict(scaledXR_test)

accuracy_train = accuracy_score(y_train1, y_pred_train)
accuracy_test = accuracy_score(y_test1, y_pred_test)
accuracy_testR = accuracy_score(y_testRe, y_pred_testR)


print("rfc on Train:", accuracy_train*100)
print("rfc on Test:", accuracy_test*100)
print("rfc on Recent Test:", accuracy_testR*100)



cnf_matrix_train = confusion_matrix(y_train1, y_pred_train)
cnf_matrix_test = confusion_matrix(y_test1, y_pred_test)
cnf_matrix_testR = confusion_matrix(y_testRe, y_pred_testR)

print(cnf_matrix_train)
print(cnf_matrix_test)
print(cnf_matrix_testR)

dump(rfc, 'rfc.joblib')
