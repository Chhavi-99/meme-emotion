from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from tensorflow.python.keras.layers import Flatten
from keras.layers.merge import concatenate
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.models import Sequential
import matplotlib as plt
from sklearn.manifold import TSNE
import pandas as pd
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
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
from keras.models import Sequential, Model
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications import VGG16, ResNet50, InceptionV3, DenseNet121
from sklearn.preprocessing import MultiLabelBinarizer
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from keras.layers import Merge
ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean_text(text):

    # Remove puncuation
    text = text.translate(string.punctuation)

    # Convert words to lower case and split them
    text = text.lower().split()

    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()
    stemmer = SnowballStemmer('english')
    stemmed_words = [stemmer.stem(word) for word in text]
    text = " ".join(stemmed_words)

    return text


df = pd.read_csv('text_train.csv', names=[
                 'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df.describe())
df['text'] = df['text'].map(lambda x: clean_text(x))
labels = []
img_data_list = []
for index, row in df.iterrows():
    if row['funny_image'] == 'funny':
        row['funny_image'] = 1
    else:
        row['funny_image'] = 0
    if row['sarcastic_image'] == 'sarcastic':
        row['sarcastic_image'] = 1
    else:
        row['sarcastic_image'] = 0
    if row['offensive_image'] == 'offensive':
        row['offensive_image'] = 1
    else:
        row['offensive_image'] = 0
    if row['motivational_image'] == 'motivational':
        row['motivational_image'] = 1
    else:
        row['motivational_image'] = 0
    labels.append([row['funny_image'], row['sarcastic_image'],
                   row['offensive_image'], row['motivational_image']])

    check = os.listdir('training_data')
    if row['image_name'] in check:
        img_path = 'training_data' + '/' + row['image_name']
    else:
        img_path = 'validation' + '/' + row['image_name']
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np.asarray(labels)
print(img_data.shape)
print(Y.shape)
# printing shape of the array

# x, y = shuffle(img_data, Y, random_state=2)
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.111, random_state=2)


vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)

embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
print(data.shape)
print(img_data.shape)
print(Y.shape)
# model
input2 = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, weights='imagenet')(input2)
model = Flatten()(model)
model = BatchNormalization()(model)
model = Dense(2048, activation='relu')(model)
model = BatchNormalization()(model)
model = Dense(1024, activation='relu')(model)
model = BatchNormalization()(model)
model = Dense(4, activation='relu')(model)
#model = Flatten()(model)
# for layer in model.layers[:-6]:
#    layer.trainable = False


input1 = Input(shape=(50,))
model_glove = Embedding(vocabulary_size, 100, input_length=50,
                        weights=[embedding_matrix])(input1)
model_glove = Dropout(0.2)(model_glove)
model_glove = Conv1D(64, 5, activation='relu')(model_glove)
model_glove = MaxPooling1D(pool_size=4)(model_glove)
model_glove = LSTM(100)(model_glove)
model_glove = Dense(4, activation='relu')(model_glove)
#model_glove = Flatten()(model_glove)

merged = concatenate([model, model_glove])
# mergedOut = Add()([model.output,model_glove.output])

# final_model = Sequential()
# final_model.add(Merge([model1, model2], mode='concat'))
final_model = Dense(4, activation='tanh', name='output_layer')(merged)
model1 = Model(inputs=[input2, input1], outputs=final_model)
model1.get_layer('resnet50').trainable = False
model1.compile(loss='hinge', optimizer='adam', metrics=['acc'])
model1.summary()
# fitting the model
start = time.time()
model1.fit([img_data, data], Y, batch_size=32, epochs=10, validation_split=0.1,
           verbose=2)

print('Training time: %s' % (start - time.time()))

# testing
# df1 = pd.read_csv('text_test.csv', names=[
#     'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# # print(df1.describe())
# df1['text'] = df1['text'].map(lambda x: clean_text(x))
# labels = []
# img_data_list = []
# for index, row in df1.iterrows():
#     if row['funny_image'] == 'funny':
#         row['funny_image'] = 1
#     else:
#         row['funny_image'] = -1
#     if row['sarcastic_image'] == 'sarcastic':
#         row['sarcastic_image'] = 1
#     else:
#         row['sarcastic_image'] = -1
#     if row['offensive_image'] == 'offensive':
#         row['offensive_image'] = 1
#     else:
#         row['offensive_image'] = -1
#     if row['motivational_image'] == 'motivational':
#         row['motivational_image'] = 1
#     else:
#         row['motivational_image'] = -1
#     labels.append([row['funny_image'], row['sarcastic_image'],
#                    row['offensive_image'], row['motivational_image']])
#     img_path = 'test' + '/' + row['image_name']
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     img_data_list.append(x)

# img_data = np.array(img_data_list)
# img_data = np.rollaxis(img_data, 1, 0)
# img_data = img_data[0]
# Y = np.asarray(labels)

# vocabulary_size = 20000
# tokenizer = Tokenizer(num_words=vocabulary_size)
# tokenizer.fit_on_texts(df1['text'])
# sequences = tokenizer.texts_to_sequences(df1['text'])
# data = pad_sequences(sequences, maxlen=50)
# # out = model_glove.evaluate(data)
# results = model1.evaluate([img_data, data], Y)
# print(results)
