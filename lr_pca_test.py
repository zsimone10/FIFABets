from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd
from keras.callbacks import ModelCheckpoint
import sys
from log_reg_network import LogReg
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, label_ranking_average_precision_score
import itertools
from sklearn.utils import resample
import math

seed = np.random.seed(7)


def load_and_resample(data_path):
    print("LOADING DATASETS")

    def load_data(path):
        data = pd.read_csv(path)
        # data = data.drop([0], axis=1)
        # data = data.drop([0], axis=0)
        return data
        # RETURN data.as_matrix()

    x = load_data(data_path)
    # print(x)
    y = load_data('data/labels_most_seasons.csv')

    ############
    # RESAMPLE
    data = pd.concat([x, y], axis=1)
    # data = data.drop([0], axis=1)

    # data = data.drop([0], axis=0)
    # print(data)
    home_wins = data[data["Home"] == 1]
    draws = data[data["Draw"] == 1]
    away_wins = data[data["Away"] == 1]
    print(len(home_wins), len(draws), len(away_wins))
    draws_resampled = resample(draws,
                               replace=True,  # sample with replacement
                               n_samples=math.ceil(len(home_wins) * 0.8),  # to half home_wins
                               random_state=123)  # reproducible results
    away_wins_resampled = resample(away_wins,
                                   replace=True,  # sample with replacement
                                   n_samples=math.ceil(len(home_wins) * 0.85),  # to match majority class
                                   random_state=123)  # reproducible results

    print(len(home_wins), len(draws_resampled), len(away_wins_resampled))
    resampled = pd.concat([draws_resampled, away_wins_resampled])
    # print(resampled)
    data = pd.concat([home_wins, resampled])

    print(x)
    y = data[["Home", "Draw", "Away"]]
    x = data.drop(["Home", "Draw", "Away"], axis=1)
    # x = x.drop(x.columns[0], axis=1)
    print(x, y)
    # y = y.drop([0], axis=0)
    # x = x.drop([0], axis=1)
    # x = x.drop([0], axis=0)
    # print(x, y)
    x, y = x.as_matrix(), y.as_matrix()
    return x, y


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
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


## multiclass or binary report
## If binary (sigmoid output), set binary parameter to True
def full_multiclass_report(model,
                           x,
                           y_true,
                           classes,
                           batch_size=32,
                           binary=False):
    # 1. Transform one-hot encoded y_true into their class number
    if not binary:
        y_true = np.argmax(y_true, axis=1)

    # 2. Predict classes and stores in y_pred
    y_pred = model.model.predict(x, batch_size=batch_size)
    y_pred = np.argmax(y_pred, axis=1)
    # 3. Print accuracy score
    print("Accuracy : " + str(accuracy_score(y_true, y_pred)))

    print("")

    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=5))

    # 5. Plot confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    print(cnf_matrix)
    plot_confusion_matrix(cnf_matrix, classes=classes)


data_paths = ['data/most_seasons_PCA_99_pct_44_components.csv'
              ]
              # 'data/most_seasons_PCA_95_pct_34_components.csv',
              # 'data/most_seasons_PCA_90_pct_27_components.csv',
              # 'data/most_seasons_PCA_85_pct_21_components.csv',
              # 'data/most_seasons_PCA_75_pct_13_components.csv'

for path in data_paths:
    x, y = load_and_resample(path)
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9, random_state=seed)

    # CLASSES DEFINE
    le = LabelEncoder()
    le.fit(["Away Win", "Draw", "Home Win"])
    print(le.classes_)

    #########################################
    # Build Network
    #
    # DETAILS:
    #
    #
    model = LogReg(x.shape[1])

    # if weights:
    #     model.load_weights(weights)
    #########################################
    # Train
    #
    #
    history = model.train(x_train, y_train)

    #########################################
    # Run Metrics
    # score = model.evaluate(x_test, y_test, batch_size=128)
    # Plot training & validation accuracy values
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Test', 'Train'], loc='upper left')
    plt.show()

    full_multiclass_report(model,
                           x_test,
                           y_test,
                           le.inverse_transform([2, 1, 0]))

