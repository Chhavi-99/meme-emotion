import sys
import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras import optimizers
from keras.layers import *
from sklearn.utils import shuffle
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.utils import np_utils
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.applications import VGG16, ResNet50, InceptionV3
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_data_list = []
labels = []
count = 0
file = open('correct_8000.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'training_data' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    if line1[1] == 'neutral':
        line1[1] = 0
    elif line1[1] == 'positive':
        line1[1] = 1
    elif line1[1] == 'very_positive':
        line1[1] = 2
    elif line1[1] == 'negative':
        line1[1] = 3
    else:
        line1[1] = 4

    labels.append(line1[1])
file.close()
# printing shape of the array

file = open('validation_data.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'validation' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    if line1[1] == 'neutral':
        line1[1] = 0
    elif line1[1] == 'positive':
        line1[1] = 1
    elif line1[1] == 'very_positive':
        line1[1] = 2
    elif line1[1] == 'negative':
        line1[1] = 3
    else:
        line1[1] = 4
    labels.append(line1[1])
file.close()

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np_utils.to_categorical(labels, 5)
print(Y.shape)
x, y = shuffle(img_data, Y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.111, random_state=0)

# model
model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
# model.add(BatchNormalization())
#model.add(Dense(2048, activation='relu'))
# model.add(BatchNormalization())
#model.add(Dense(1024, activation='relu'))
# model.add(BatchNormalization())
model.add(Dense(5, activation='softmax'))
for layer in model.layers[:-3]:
    layer.trainable = False
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])
model.summary()
# fitting the model
start = time.time()
history = model.fit(X_train, y_train, batch_size=32, epochs=10,
                    verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (start - time.time()))

# model.evaluate
img_data_list1 = []
labels1 = []
count1 = 0
file = open('test_data.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'test' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list1.append(x)
    if line1[1] == 'neutral':
        line1[1] = 0
    elif line1[1] == 'positive':
        line1[1] = 1
    elif line1[1] == 'very_positive':
        line1[1] = 2
    elif line1[1] == 'negative':
        line1[1] = 3
    else:
        line1[1] = 4
    labels1.append(line1[1])
file.close()
# printing shape of the array
img_data = np.array(img_data_list1)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np_utils.to_categorical(labels1, 5)
results = model.evaluate(img_data, Y, batch_size=64)
print(results)

image_list = os.listdir('test')
file = open('test_output5.csv', 'a')
for imag in image_list:
    img_path = 'test' + '/' + imag
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    f = ''
    if preds[0][0] >= 0.5:
        f = 'neutral'
    elif preds[0][1] >= 0.5:
        f = 'positive'
    elif preds[0][2] >= 0.5:
        f = 'very_positive'
    elif preds[0][3] >= 0.5:
        f = 'negative'
    else:
        f = 'very_negative'
    file.write(imag + ',' + f)
    file.write('\n')
file.close()
