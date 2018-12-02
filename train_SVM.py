from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers#, uti
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
from sklearn.utils import resample
import math
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from joblib import dump, load


seed = np.random.seed(7)

print("LOADING DATASETS")
def load_data(path):
    data = pd.read_csv(path)
    #data = data.drop([0], axis=1)
    #data = data.drop([0], axis=0)
    return data
    #RETURN data.as_matrix()
x = load_data('data/most_seasons_unnormalized.csv')
#print(x)
y = load_data('data/labels_most_seasons.csv')

x1 = load_data('data/labels_recent_seasons.csv')
#print(x)
y1 = load_data('data/recent_seasons_unnormalized.csv')

def parseData(x, y, c):
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

x_train, x_test, y_train, y_test = parseData(x,y, 0.9)
x_trainR, x_testR, y_trainR, y_testR = parseData(x1,y1, 0.0)

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

#SVM Linear Model


#svm = SVC(kernel='linear', C=1, probability=False,random_state=0, verbose = False, tol=0.001, cache_size=1, max_iter = 100)
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
param_grid = {'C': Cs, 'gamma' : gammas}
#for C in Cs:
#    for gamma in gammas:
svm = SVC(kernel='rbf', probability=False, verbose = False, tol= 1e-6, cache_size=10000, C = 1)

#all_accuracies = cross_val_score(estimator=svm, X=scaledX_train, y=y_train1, cv=5)

#print(all_accuracies)
#print(all_accuracies.mean())
#print(all_accuracies.std())

svm.fit(scaledX_train, y_train1)


y_pred = svm.predict(scaledX_test)
#y_prob = svm.predict_proba(scaledX_test)
y_pred_train = dt.predict(scaledX_train)
y_pred_test = dt.predict(scaledX_test)
y_pred_testR = dt.predict(scaledXR_test)

accuracy_train = accuracy_score(y_train1, y_pred_train)
accuracy_test = accuracy_score(y_test1, y_pred_test)
accuracy_testR = accuracy_score(y_testRe, y_pred_testR)


print("rbf on Train:", accuracy_train*100)
print("rbf on Test:", accuracy_test*100)
print("rbf on Recent Test:", accuracy_testR*100)

cnf_matrix_train = confusion_matrix(y_train1, y_pred_train)
cnf_matrix_test = confusion_matrix(y_test1, y_pred_test)
cnf_matrix_testR = confusion_matrix(y_testRe, y_pred_testR)


#Cs = [0.001, 0.01, 0.1, 1, 10]
#gammas = [0.001, 0.01, 0.1, 1]
#param_grid = {'C': Cs, 'gamma' : gammas}
#grid_search = GridSearchCV(svm, param_grid, cv=2)
#grid_search.fit(scaledX_train, y_train1)
#print (grid_search.best_params_)

#best_result = grid_search.best_score_
#print('Best C:',grid_search.best_estimator_.C)
#print('Best Gamma:', grid_search.best_estimator_.gamma)
#print(best_result)




#print("rbf")
def plot_confusion_matrix(cm, classes, normalize=False,cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix'

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()


plot_confusion_matrix(cnf_matrix_train, classes = ["Home", "Draw", "Away"])
plot_confusion_matrix(cnf_matrix_test, classes = ["Home", "Draw", "Away"])
plot_confusion_matrix(cnf_matrix_testR, classes = ["Home", "Draw", "Away"])


dump(svm, 'SVM.joblib')

#print(classification_report(y_test1, y_pred))
