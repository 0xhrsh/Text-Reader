import pandas as pd 
import numpy as np 
import cv2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as k 
from keras.utils import np_utils
import tensorflow as tf 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

tf.logging.set_verbosity(tf.logging.ERROR)
print("Reading Data")
dataset=pd.read_csv("data.csv").astype('float32')
dataset.rename(columns={'0':'label'},inplace=True)
X=dataset.drop('label',axis=1)
y=dataset['label']
#print("shape",X.shape)
#print(y.shape,len(X.iloc[1]))
print("Preprocessing Data")
X_train, X_test, y_train, y_test=train_test_split(X,y)
standard_scale=MinMaxScaler()
standard_scale.fit(X_train)
X_train=standard_scale.transform(X_train)
X_test=standard_scale.transform(X_test)
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
y_train=np_utils.to_categorical(y_train)
y_test=np_utils.to_categorical(y_test)

print("Building the model..")
model=Sequential()
model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(26,activation="softmax"))
print("Compiling...")
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("Training...")
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=4096, verbose=1)
print("evaluating")
scores = model.evaluate(X_test,y_test, verbose=0)
print("CNN accuracy: %.2f%%" % (scores[1]*100))
print("Saving Model")
model.save('weights.h5')
model.save_weights("model.h5")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
