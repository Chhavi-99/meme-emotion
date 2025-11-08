from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import matplotlib as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import Flatten
from keras.utils import np_utils

# Plot
# import plotly.offline as py
# import plotly.graph_objs as go


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


df = pd.read_csv('text_train_new.csv', names=[
                 'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df.describe())
df['text'] = df['text'].map(lambda x: clean_text(x))
label1 = []
label2 = []
label3 = []
label4 = []
for index, row in df.iterrows():
    if row['funny_image'] == 'not_funny':
        row['funny_image'] = 0
    elif row['funny_image'] == 'funny':
        row['funny_image'] = 1
    elif row['funny_image'] == 'very_funny':
        row['funny_image'] = 2
    elif row['funny_image'] == 'hilarious':
        row['funny_image'] = 3
    label1.append(row['funny_image'])

    if row['sarcastic_image'] == 'not_sarcastic':
        row['sarcastic_image'] = 0
    elif row['sarcastic_image'] == 'general':
        row['sarcastic_image'] = 1
    elif row['sarcastic_image'] == 'twisted_meaning':
        row['sarcastic_image'] = 2
    elif row['sarcastic_image'] == 'very_twisted':
        row['sarcastic_image'] = 3
    label2.append(row['sarcastic_image'])

    if row['offensive_image'] == 'not_offensive':
        row['offensive_image'] = 0
    elif row['offensive_image'] == 'slight':
        row['offensive_image'] = 1
    elif row['offensive_image'] == 'very_offensive':
        row['offensive_image'] = 2
    elif row['offensive_image'] == 'hateful_offensive':
        row['offensive_image'] = 3
    label3.append(row['offensive_image'])

    if row['motivational_image'] == 'motivational':
        row['motivational_image'] = 0
    elif row['motivational_image'] == 'not_motivational':
        row['motivational_image'] = 1
    label4.append(row['motivational_image'])

Y1 = np_utils.to_categorical(label1, 4)
Y2 = np_utils.to_categorical(label2, 4)
Y3 = np_utils.to_categorical(label3, 4)
Y4 = np_utils.to_categorical(label4, 2)
# print(labels)
# tokenization
vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)

# model_glove = Conv1D(64, 5, activation='relu')(model_glove)
# model_glove = MaxPooling1D(pool_size=4)(model_glove)curacy'])

# model_lstm.fit(data, np.array(labels), validation_split=0.1, epochs=10)
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

input1 = Input(shape=(50,))
model_glove = Embedding(vocabulary_size, 100, input_length=50,
                        weights=[embedding_matrix], trainable=False)(input1)
model_glove = Dropout(0.2)(model_glove)
model_glove = Conv1D(64, 5, activation='relu')(model_glove)
model_glove = MaxPooling1D(pool_size=4)(model_glove)
model_glove = LSTM(100)(model_glove)
out1 = Dense(4, activation='softmax')(model_glove)
out2 = Dense(4, activation='softmax')(model_glove)
out3 = Dense(4, activation='softmax')(model_glove)
out4 = Dense(2, activation='softmax')(model_glove)
model = Model(inputs=input1, outputs=[out1, out2, out3, out4])
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(data, [Y1, Y2, Y3, Y4], validation_split=0.1, epochs=10, verbose=2)

df = pd.read_csv('text_test_new.csv', names=[
                 'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df.describe())
df['text'] = df['text'].map(lambda x: clean_text(x))
label1 = []
label2 = []
label3 = []
label4 = []
for index, row in df.iterrows():
    if row['funny_image'] == 'not_funny':
        row['funny_image'] = 0
    elif row['funny_image'] == 'funny':
        row['funny_image'] = 1
    elif row['funny_image'] == 'very_funny':
        row['funny_image'] = 2
    elif row['funny_image'] == 'hilarious':
        row['funny_image'] = 3
    label1.append(row['funny_image'])

    if row['sarcastic_image'] == 'not_sarcastic':
        row['sarcastic_image'] = 0
    elif row['sarcastic_image'] == 'general':
        row['sarcastic_image'] = 1
    elif row['sarcastic_image'] == 'twisted_meaning':
        row['sarcastic_image'] = 2
    elif row['sarcastic_image'] == 'very_twisted':
        row['sarcastic_image'] = 3
    label2.append(row['sarcastic_image'])

    if row['offensive_image'] == 'not_offensive':
        row['offensive_image'] = 0
    elif row['offensive_image'] == 'slight':
        row['offensive_image'] = 1
    elif row['offensive_image'] == 'very_offensive':
        row['offensive_image'] = 2
    elif row['offensive_image'] == 'hateful_offensive':
        row['offensive_image'] = 3
    label3.append(row['offensive_image'])

    if row['motivational_image'] == 'motivational':
        row['motivational_image'] = 0
    elif row['motivational_image'] == 'not_motivational':
        row['motivational_image'] = 1
    label4.append(row['motivational_image'])

Y1 = np_utils.to_categorical(label1, 4)
Y2 = np_utils.to_categorical(label2, 4)
Y3 = np_utils.to_categorical(label3, 4)
Y4 = np_utils.to_categorical(label4, 2)
# print(labels)
# tokenization
# vocabulary_size = 20000
# tokenizer = Tokenizer(num_words=vocabulary_size)
# tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)
#print(data)
results=model.evaluate(data,[Y1,Y2,Y3,Y4])
print(results)
file1=open('abc.txt','a')
final=model.predict(data)
for line in final:
    for line1 in line:
        print(line1)
#             file1.write(line3[0]+','+line3[1]+','+line3[2]+','+line3[3])
#             file1.write('\n')
file1.close()
            
# file1=open('abc.txt','a')
# f
# file1.close()
#file1=open('text_val.csv','a')
#file=open('text_test_new.csv','r')
#file2=open('text_prob.csv','a')sequences = tokenizer.texts_to_sequences(df['text'])
# data = pad_sequences(sequences, maxlen=50)
# for line in file:
#     line1=line.rstrip('\r\n').split(',')
#     sequences = tokenizer.texts_to_sequences(line1[-1])
#     print(sequences)
#     data = pad_sequences(sequences, maxlen=50)
#     print(data)
#     print("------------------------------------------")
#     preds=model.predict(data)
#     #print(preds)
#     f = ''
#     g = ''
#     h = ''
#     i = ''
#     class1 = np.argmax(preds[0])
#     if class1 == 0:
#         f = 'not_funny'
#     elif class1 == 1:
#         f = 'funny'
#     elif class1 == 2:
#         f = 'very_funny'
#     elif class1 == 3:
#         f = 'hilarious'
#     class2 = np.argmax(preds[1])
#     if class2 == 0:
#         g = 'not_sarcastic'
#     elif class2 == 1:
#         g = 'general'
#     elif class2 == 2:
#         g = 'twisted_meaning'
#     elif class2 == 3:
#         g = 'very_twisted'
#     class3 = np.argmax(preds[2])
#     if class3 == 0:
#         h = 'not_offensive'
#     elif class3 == 1:
#         h = 'slight'
#     elif class3 == 2:
#         h = 'very_offensive'
#     elif class3 == 3:
#         h = 'hateful_offensive'
#     class4 = np.argmax(preds[3])
#     if class4 == 0:
#         i = 'motivational'
#     elif class4 == 1:
#         i = 'not_motivational'

# #     # file1.write(f + ',' + g + ',' + h + ',' + i)
# #     # file1.write('\n')
# #     # file2.write(str(np.max(preds[0])) +
# #     #             ',' + str(np.max(preds[1])) + ',' + str(np.max(preds[2])) + ',' + str(np.max(preds[3])))
# #     # file2.write('\n')
# # file1.close()
# # file2.close()
# file.close()
