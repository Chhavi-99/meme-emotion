from sklearn.metrics import confusion_matrix
from sklearn.metrics import *
from tensorflow.python.keras.layers.merge import concatenate
import matplotlib as plt
#from sklearn.manifold import TSNE
import pandas as pd
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import sys
import os
import time
import zipfile
import numpy as np
import tensorflow as tf
from PIL import ImageFile
from sklearn.utils import shuffle
from tensorflow.python.keras.preprocessing import image
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import np_utils
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.applications import *
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.initializers import *
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


df = pd.read_csv('label2.csv', names=[
                 'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df.describe())
#df['text'] = df.fillna({'text': ''})
df['text'] = df['text'].map(lambda x: clean_text(x))
print(df.head())
labels = []
img_data_list = []
#check = os.listdir('training_data')

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

    # if row['image_name'] in check:
    #     img_path = 'training_data' + '/' + row['image_name']
    #     img = image.load_img(img_path, target_size=(224, 224))
    # else:
    #     img_path = 'validation' + '/' + row['image_name']
    #     img = image.load_img(img_path, target_size=(224, 224))
    img_path = 'images' + '/' + row['image_name']
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list.append(x)

img_data = np.array(img_data_list)
img_data = np.rollaxis(img_data, 1, 0)
img_data = img_data[0]
Y = np.array(labels)
print(img_data.shape)
print(Y.shape)
print(df.shape)

labels_1 = []
img_data_list_1 = []

df1 = pd.read_csv('trial2.csv', names=[
    'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
df1['text'] = df1['text'].map(lambda x: clean_text(x))

for index, row in df1.iterrows():
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
    labels_1.append([row['funny_image'], row['sarcastic_image'],
                     row['offensive_image'], row['motivational_image']])
    img_path = 'trial/Meme_images' + '/' + row['image_name']
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    img_data_list_1.append(x)

img_data_1 = np.array(img_data_list_1)
img_data_1 = np.rollaxis(img_data_1, 1, 0)
img_data_1 = img_data_1[0]
Y_1 = np.array(labels_1)
print(img_data_1.shape)
print(Y_1.shape)
print(df1.shape)


# printing shape of the array

# x, y = shuffle(img_data, Y, random_state=2)
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.111, random_state=2)

tokenizer = Tokenizer()
total = df['text'].values+df1['text'].values

tokenizer.fit_on_texts(total)
vocabulary_size = len(tokenizer.word_index)+1
max_length = max([len(s.split()) for s in df['text']])

# print(tokenizer.word_index)

sequences = tokenizer.texts_to_sequences(df['text'])
# print(sequences)
data = pad_sequences(sequences, maxlen=max_length, padding='post')

validation = tokenizer.texts_to_sequences(df1['text'])
validation_data = pad_sequences(validation, maxlen=max_length, padding='post')


embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    #print(word, coefs)
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
# # model
# input2 = Input(shape=(224, 224, 3))
# model = VGG16(include_top=True, weights='imagenet')(input2)
# model = Flatten()(model)
# model = BatchNormalization()(model)
# model = Dense(2048, activation='relu')(model)
# model = BatchNormalization()(model)
# model = Dense(1024, activation='relu')(model)
# model = BatchNormalization()(model)
# model = Dense(4, activation='relu')(model)
# #model = Flatten()(model)
# # for layer in model.layers[:-6]:
# #    layer.trainable = False


# input1 = Input(shape=(50,))
# model_glove = Embedding(vocabulary_size, 100, input_length=50,
#                         weights=[embedding_matrix], trainable=False)(input1)
# #model_glove = Dropout(0.2)(model_glove)
# model_glove = Conv1D(64, 5, activation='relu')(model_glove)
# #model_glove = MaxPooling1D(pool_size=4)(model_glove)
# model_glove = Flatten()(model_glove)
# #model_glove = LSTM(100)(model_glove)
# #model_glove = Flatten()(model_glove)
# #model_glove = Dense(4, activation='relu')(model_glove)
# #model_glove = Flatten()(model_glove)

# merged = concatenate([model, model_glove])
# merged = Dense(2048, activation='relu')(merged)
# merged = Dense(1024, activation='relu')(merged)
# # mergedOut = Add()([model.output,model_glove.output])

# model_vgg = VGG16(include_top=True, weights='imagenet')
# for layer in model_vgg.layers[:-5]:
#     layer.trainable = False

# model1
input1 = Input(shape=(224, 224, 3))
model = VGG16(include_top=True, weights='imagenet')(input1)
model = Dense(256, activation='relu')(model)


input2 = Input(shape=(max_length,))
model_glove = Embedding(vocabulary_size, 100, input_length=max_length,
                        embeddings_initializer=Constant(embedding_matrix), trainable=False)(input2)
model_glove = LSTM(100)(model_glove)
model_glove = Dense(256, activation='relu')(model_glove)


merged = concatenate([model, model_glove])
final_model = Dense(4, activation='sigmoid', name='output_layer')(merged)
model1 = Model(inputs=[input1, input2], outputs=final_model)
model1.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])

model1.summary()
print(model1.layers[3].name)
for layer in model1.layers[3].layers[:-5]:
    layer.trainable = False

es = EarlyStopping(monitor='val_acc', mode='auto',
                   restore_best_weights=True, verbose=1, patience=6)
model1.fit([img_data, data], Y, batch_size=32, validation_data=([img_data_1,validation_data],Y_1),callbacks=[
           es], epochs=20)


# for layer in model1:
#     new_model.layers[2].get_layer('bn5c_branch2c').trainable
# final_model = Sequential()
# final_model.add(Merge([model1, model2], mode='concat'))
# final_model = Dense(4, activation='sigmoid')(merged)
# model1 = Model(inputs=[input2, input1], outputs=final_model)
# model1.get_layer('vgg16').trainable = False
# model1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model1.summary()
# fitting the model
# es = EarlyStopping(monitor='val_acc', mode='auto',
#                    restore_best_weights=True, verbose=1, patience=6)
# model1.fit([img_data, data], Y, batch_size=32, callbacks=[
#            es], epochs=20, validation_split=0.1)
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
# #
df = pd.read_csv('task2_test.csv', names=[
                 'image_name', 'text'])
file = open('task2_output.csv', 'a')
# df['text'] = df['text'].map(lambda x: clean_text(x))
# image_name = []
# for index, row in df.iterrows():
#     image_name.append(row['image_name'])

for index, row in df.iterrows():
    img_path = '2000_data/' + row['image_name']
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    sequences = tokenizer.texts_to_sequences(row['text'])
    data = pad_sequences(sequences, maxlen=max_length, padding='post')
    preds = model.predict([x, data])
    val1 = ''
    val2 = ''
    val3 = ''
    val4 = ''
    if preds[0][0] >= 0.5:
        val1 = 'funny'
    else:
        val1 = 'not_funny'
    if preds[0][1] >= 0.5:
        val2 = 'sarcastic'
    else:
        val2 = 'not_sarcastic'
    if preds[0][2] >= 0.5:
        val3 = 'offensive'
    else:
        val3 = 'not_offensive'
    if preds[0][1] >= 0.5:
        val4 = 'motivational'
    else:
        val4 = 'not_motivational'
    file.write(row['image_name']+','+val1+','+val2+','+val3+','+val4)
    file.write('\n')
file.close()
