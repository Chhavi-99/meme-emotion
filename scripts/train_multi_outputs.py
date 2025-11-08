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
label1 = []
label2 = []
label3 = []
label4 = []
count = 0
file = open('correct_8000_1.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    check1 = os.listdir('training_data')
    if line1[0] in check1:
        img_path = 'training_data' + '/' + line1[0]
    else:
        img_path = 'validation' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    if line1[1] == 'not_funny':
        line1[1] = 0
    elif line1[1] == 'funny':
        line1[1] = 1
    elif line1[1] == 'very_funny':
        line1[1] = 2
    elif line1[1] == 'hilarious':
        line1[1] = 3
    label1.append(line1[1])

    if line1[2] == 'not_sarcastic':
        line1[2] = 0
    elif line1[2] == 'general':
        line1[2] = 1
    elif line1[2] == 'twisted_meaning':
        line1[2] = 2
    elif line1[2] == 'very_twisted':
        line1[2] = 3
    label2.append(line1[2])

    if line1[3] == 'not_offensive':
        line1[3] = 0
    elif line1[3] == 'slight':
        line1[3] = 1
    elif line1[3] == 'very_offensive':
        line1[3] = 2
    elif line1[3] == 'hateful_offensive':
        line1[3] = 3
    label3.append(line1[3])

    if line1[4] == 'motivational':
        line1[4] = 0
    elif line1[4] == 'not_motivational':
        line1[4] = 1
    label4.append(line1[4])
file.close()
# printing shape of the array
# print(label2)

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y1 = np_utils.to_categorical(label1, 4)
Y2 = np_utils.to_categorical(label2, 4)
Y3 = np_utils.to_categorical(label3, 4)
Y4 = np_utils.to_categorical(label4, 2)
# print(Y.shape)
# x, y = shuffle(img_data, Y, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.111, random_state=0)

# model
input1 = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, pooling='avg', weights='imagenet')(input1)
model = Flatten()(model)
model = Dense(256, activation='relu')(model)
out1 = Dense(4, activation="softmax")(model)
out2 = Dense(4, activation="softmax")(model)
out3 = Dense(4, activation="softmax")(model)
out4 = Dense(2, activation="softmax")(model)
model1 = Model(inputs=input1, outputs=[out1, out2, out3, out4])
for layer in model1.layers[:-3]:
    layer.trainable = False
model1.compile(loss='categorical_crossentropy',
               optimizer='adam', metrics=['acc'])

# fitting the model
start = time.time()
history = model1.fit(img_data, [Y1, Y2, Y3, Y4], batch_size=32, epochs=10,
                     verbose=2, validation_split=0.111)
print('Training time: %s' % (start - time.time()))

img_data_list1 = []
label1 = []
label2 = []
label3 = []
label4 = []
count1 = 0
file = open('test_data1.csv', 'r')
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'test' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list1.append(x)
    if line1[1] == 'not_funny':
        line1[1] = 0
    elif line1[1] == 'funny':
        line1[1] = 1
    elif line1[1] == 'very_funny':
        line1[1] = 2
    elif line1[1] == 'hilarious':
        line1[1] = 3
    label1.append(line1[1])

    if line1[2] == 'not_sarcastic':
        line1[2] = 0
    elif line1[2] == 'general':
        line1[2] = 1
    elif line1[2] == 'twisted_meaning':
        line1[2] = 2
    elif line1[2] == 'very_twisted':
        line1[2] = 3
    label2.append(line1[2])

    if line1[3] == 'not_offensive':
        line1[3] = 0
    elif line1[3] == 'slight':
        line1[3] = 1
    elif line1[3] == 'very_offensive':
        line1[3] = 2
    elif line1[3] == 'hateful_offensive':
        line1[3] = 3
    label3.append(line1[3])

    if line1[4] == 'motivational':
        line1[4] = 0
    elif line1[4] == 'not_motivational':
        line1[4] = 1
    label4.append(line1[4])
file.close()
# printing shape of the array
img_data = np.array(img_data_list1)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y1 = np_utils.to_categorical(label1, 4)
Y2 = np_utils.to_categorical(label2, 4)
Y3 = np_utils.to_categorical(label3, 4)
Y4 = np_utils.to_categorical(label4, 2)
results = model1.evaluate(img_data, [Y1, Y2, Y3, Y4], batch_size=32)
print(results)

image_list = os.listdir('test')
file = open('test_output6.csv', 'a')
file1 = open('test_output7.csv', 'a')
for imag in image_list:
    img_path = 'test' + '/' + imag
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model1.predict(x)
    f = ''
    g = ''
    h = ''
    i = ''
    class1 = np.argmax(preds[0])
    if class1 == 0:
        f = 'not_funny'
    elif class1 == 1:
        f = 'funny'
    elif class1 == 2:
        f = 'very_funny'
    elif class1 == 3:
        f = 'hilarious'
    class2 = np.argmax(preds[1])
    if class2 == 0:
        g = 'not_sarcastic'
    elif class2 == 1:
        g = 'general'
    elif class2 == 2:
        g = 'twisted_meaning'
    elif class2 == 3:
        g = 'very_twisted'
    class3 = np.argmax(preds[2])
    if class3 == 0:
        h = 'not_offensive'
    elif class3 == 1:
        h = 'slight'
    elif class3 == 2:
        h = 'very_offensive'
    elif class3 == 3:
        h = 'hateful_offensive'
    class4 = np.argmax(preds[3])
    if class4 == 0:
        i = 'motivational'
    elif class4 == 1:
        i = 'not_motivational'

    file.write(imag + ',' + f + ',' + g + ',' + h + ',' + i)
    file.write('\n')
    file1.write(imag + ',' + str(np.max(preds[0])) +
                ',' + str(np.max(preds[1])) + ',' + str(np.max(preds[2])) + ',' + str(np.max(preds[3])))
    file1.write('\n')
file.close()
file1.close()
