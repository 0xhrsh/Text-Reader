import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import cv2
import math
import json
from pprint import pprint
from keras.preprocessing.image import ImageDataGenerator
import glob
from keras.utils import to_categorical
model_json=open('model.json','r')
loaded_model_json=model_json.read()
model_json.close()
loaded_model=model_from_json(loaded_model_json)
loaded_model.load_weights("weights.h5")
print("model loaded from dics")
img=cv2.imread("test_5.jpg")
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(gray_image,(28,28))
#cv2.imshow("img",img)
print(img.shape)
#cv2.waitKey(0)
img=np.expand_dims(img, axis=0)
print(img.shape)
img=gray_image/255
loaded_model.predict(img)