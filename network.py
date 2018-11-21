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
    def full_multiclass_report(self,
                               x,
                               y_true,
                               classes,
                               batch_size=32,
                               binary=False):
        # 1. Transform one-hot encoded y_true into their class number
        if not binary:
            y_true = np.argmax(y_true, axis=1)

        # 2. Predict classes and stores in y_pred
        y_pred = self.model.model.predict(x, batch_size=batch_size)
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

    # full_multiclass_report(self.model,
    #                        x_test,
    #                        y_test,
    #                        le.inverse_transform(np.arange(3)))

    def eval(self, x_test, y_test, le):
        self.full_multiclass_report(
                       x_test,
                       y_test,
                       le.inverse_transform(np.arange(3)))


