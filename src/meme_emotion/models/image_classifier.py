#from sklearn.svm import SVC
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
import sys
import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from keras import optimizers
from keras.layers import *
#from sklearn.utils import shuffle
from keras.preprocessing import image
from matplotlib import pyplot as plt
from keras.utils import np_utils
#from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
#from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from tensorflow.python.keras.callbacks import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix
# from sklearn.preprocessing import MultiLabelBinarizer
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.ensemble import GradientBoostingClassifier
ImageFile.LOAD_TRUNCATED_IMAGES = True

img_data_list = []
labels = []
count = 0
file = open('meme_images.csv', 'r')
count = 0
for line in file:
    line1 = line.rstrip('\r\n').split(',')
    img_path = 'training_data' + '/' + line1[0]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    labels.append(1)
    if count == 3500:
        break
    count += 1
file.close()

count = 0
list1 = os.listdir('embedded_text/embedded_text/')
for img in list1:
    img_path = 'embedded_text/embedded_text/'+img
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)
    labels.append(0)
    if count == 3500:
        break
    count += 1


img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
print(img_data.shape)
Y = np.asarray(labels)
print(Y.shape)
#x, y = shuffle(img_data, Y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    img_data, Y, test_size=0.1, random_state=42)


# model
model = Sequential()
model.add(VGG16(include_top=True, weights='imagenet'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['acc'])
model.summary()
for layer in model.layers[:-5]:
    layer.trainable = False

time.time()
es = EarlyStopping(monitor='val_acc', mode='auto',
                   restore_best_weights=True, verbose=1, patience=6)
model.fit(X_train, y_train, batch_size=64,
          epochs=1, validation_split=0.1, callbacks=[es])
print('Training time: %s' % (start - time.time()))

yhat_probs = model.predict(X_test, verbose=0)
yhat_classes = model.predict_classes(X_test, verbose=0)
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]

accuracy = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, yhat_classes)
print('Precision: %f' % precision)
recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % recall)
f1 = f1_score(y_test, yhat_classes)
print('F1 score: %f' % f1)

print('**************************')
cm = confusion_matrix(y_test, yhat_classes)
print(cm)