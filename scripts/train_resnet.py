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
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
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
file = open('meme_images.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'training_data' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    if line1[1] == 'funny':
        line1[1] = 1
    else:
        line1[1] = 0
    if line1[2] == 'sarcastic':
        line1[2] = 1
    else:
        line1[2] = 0
    if line1[3] == 'offensive':
        line1[3] = 1
    else:
        line1[3] = 0
    if line1[4] == 'motivational':
        line1[4] = 1
    else:
        line1[4] = 0
    labels.append([line1[1], line1[2], line1[3], line1[4]])
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
    if line1[1] == 'funny':
        line1[1] = 1
    else:
        line1[1] = 0
    if line1[2] == 'sarcastic':
        line1[2] = 1
    else:
        line1[2] = 0
    if line1[3] == 'offensive':
        line1[3] = 1
    else:
        line1[3] = 0
    if line1[4] == 'motivational':
        line1[4] = 1
    else:
        line1[4] = 0
    labels.append([line1[1], line1[2], line1[3], line1[4]])
file.close()

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np.asarray(labels)
print(Y.shape)
x, y = shuffle(img_data, Y, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.111, random_state=2)


# model
model = Sequential()
model.add(VGG16(include_top=False, pooling='avg', weights='imagenet'))
model.add(Flatten())
model.add(BatchNormalization())
# model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(4, activation='sigmoid'))

model.get_layer('vgg16').trainable = False

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])
model.summary()
# fitting the model
start = time.time()
history = model.fit(X_train, y_train, batch_size=32, epochs=10,
                    verbose=2, validation_data=(X_test, y_test))

print('Training time: %s' % (start - time.time()))

# testing
# image_list = os.listdir('test')
# file = open('test_output4.csv', 'a')
# for imag in image_list:
#     img_path = 'test' + '/' + imag
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     preds = model.predict(x)
#     # if preds[0][0] >= 0.5:
#     #     f = 'funny'
#     # else:
#     #     f = 'not_funny'
#     # if preds[0][1] >= 0.5:
#     #     m = 'sarcastic'
#     # else:
#     #     m = 'not_sarcastic'
#     # if preds[0][2] >= 0.5:
#     #     o = 'offensive'
#     # else:
#     #     o = 'not_offensive'
#     # if preds[0][3] >= 0.5:
#     #     s = 'motivational'
#     # else:
#     #     s = 'not_motivational'
#     file.write(imag + ',' + str(preds[0][0]) + ',' + str(preds[0]
#                                                          [1]) + ',' + str(preds[0][2]) + ',' + str(preds[0][3]))
#     file.write('\n')
# file.close()

# # model.evaluate
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
    if line1[1] == 'funny':
        line1[1] = 1
    else:
        line1[1] = 0
    if line1[2] == 'sarcastic':
        line1[2] = 1
    else:
        line1[2] = 0
    if line1[3] == 'offensive':
        line1[3] = 1
    else:
        line1[3] = 0
    if line1[4] == 'motivational':
        line1[4] = 1
    else:
        line1[4] = 0
    labels1.append([line1[1], line1[2], line1[3], line1[4]])
file.close()
# printing shape of the array
img_data = np.array(img_data_list1)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np.asarray(labels1)
results = model.evaluate(img_data, Y, batch_size=64)
print(results)
