import pandas as pd 
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.preprocessing import image
from keras.optimizers import RMSprop
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Lambda, Flatten
from keras.utils.np_utils import to_categorical
from keras import backend as K
import math

BATCH_SIZE = 64
EPOCHS = 10

#prepare training set 
train = pd.read_csv('C:/Users/Coko/PythonProjects/train.csv') #open train.csv file
X_train = (train.ix[:,1:].values).astype('float32') 
#format the training set in shape of (number of images, rows of img, cols of img, 1 for gray colour channel)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1) 
y_train = (train.ix[:, 0].values).astype('int32')
y_train = to_categorical(y_train) #convert the value of the label into a 10-digit vector 
#with 0s except for the position of this number which is 1. e.g 4 --> [0,0,0,0,1,0,0,0,0,0]

#split the training sest into training and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.30, random_state=42) 

test = pd.read_csv('C:/Users/Coko/PythonProjects/test.csv') #open test.csv file
X_test = (test.values.astype('float32'))
X_test = X_test.reshape(X_test.shape[0], 28, 28,1) #same format for test set as for training set 

mean_px = X_train.mean().astype('float32')
std_px = X_train.std().astype('float32')

#function to centre the number in image
def standard(x): 
    return (x-mean_px)/std_px

#Building the model using Keras
model = Sequential()
model.add(Conv2D(filters = 10, kernel_size = 7, activation = 'tanh', input_shape=(28,28,1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])

model.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

generator = image.ImageDataGenerator()

train_generator = generator.flow(X_train,y_train, batch_size=BATCH_SIZE)
val_generator = generator.flow(X_val,y_val, batch_size=BATCH_SIZE)


model.fit_generator(train_generator, steps_per_epoch = 460, epochs=5, \
    validation_data=val_generator, validation_steps=460)

score = model.evaluate(X_val, y_val, batch_size=32)
print(model.metrics_names[1], score[1]*100)

predictions = model.predict_classes(X_test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)), "Label": predictions})
submissions.to_csv("C:/Users/Coko/PythonProjects/DR_new.csv", index=False, header=True) #save the predicted numbers in DR_new.csv file 
