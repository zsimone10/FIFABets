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
# numEpochs = 10000
# if len(sys.argv) > 1:
#     numEpochs = sys.argv[1]


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
ext = []
for j in range(0,2):
    ext.append(np.transpose(x[j]))
Xnew = np.asarray(ext)
print(Xnew, Xnew.shape)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(0, len(ynew)):
    print("X=%s, Predicted=%s" % (Xnew, ynew[i]))
#########################################
#Train
#
#
print("TRAINING...")
filepath="weights/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=500)

callbacks_list = [checkpoint]
history = model.fit(x, y,
        validation_split=0.1, #initially .25
        epochs=2000,
        batch_size=10,
        callbacks = callbacks_list)

ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(0, len(ynew)):
    print("X=%s, Predicted=%s" % (Xnew, ynew[i]))

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
