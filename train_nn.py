from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers, utils
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pandas as pd


np.random.seed(7)

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
# x = np.random.random((1000, 80))
# y = utils.to_categorical(np.random.randint(2, size=(1000, 1)), num_classes=3)
#x_test = np.random.random((100, 80))
#y_test = utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=3)


#########################################
#Build Network
#
#DETAILS:
#
#
print("BUILDING NETWORK...")

model = Sequential()
model.add(Dense(512, input_dim=78))
model.add(Activation('sigmoid'))
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(64))
model.add(Activation('sigmoid'))
model.add(Dense(3))
model.add(Activation('softmax'))


sgd = optimizers.SGD(lr=0.09,)
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])
plot_model(model, to_file='model.png')
print(model.summary())
#########################################
#Train
#
#
print("TRAINING...")
history = model.fit(x, y,
        validation_split=0.25, #initially .25
        epochs=10000,
        batch_size=10)

#########################################
#Run Metrics
#score = model.evaluate(x_test, y_test, batch_size=128)
# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


#########################################
#Save Weights
#


