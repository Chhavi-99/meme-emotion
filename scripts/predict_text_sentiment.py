from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import string
import re
from sklearn.utils import shuffle
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
from keras.optimizers import SGD
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


df = pd.read_csv('train_text_senti.csv', names=[
                 'text', 'sentiment'])
# print(df.describe())
df['text'] = df['text'].map(lambda x: clean_text(x))
labels = []
for index, row in df.iterrows():
    print(row['sentiment'])
    if row['sentiment'] == 'neutral':
        row['sentiment'] = 0
    elif row['sentiment'] == 'positive':
        row['sentiment'] = 1
    elif row['sentiment'] == 'very_positive':
        row['sentiment'] = 1
    elif row['sentiment'] == 'negative':
        row['sentiment'] = 2
    else:
        row['sentiment'] = 2
    labels.append(row['sentiment'])
Y = np_utils.to_categorical(labels, 3)
# print(Y)
# print(df)
# print(labels)
# tokenization
vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=50)
print(data.shape)
print(Y.shape)
# x, y = shuffle(data, Y, random_state=2)
# X_train, X_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.111, random_state=2)
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
model_glove = Dense(3, activation='softmax')(model_glove)
model = Model(inputs=input1, outputs=model_glove)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
model.summary()
model.fit(data, Y, validation_split=0.1, epochs=10, verbose=2)

df1 = pd.read_csv('test_text_senti.csv', names=[
    'text', 'sentiment'])
# print(df1.describe())
df1['text'] = df1['text'].map(lambda x: clean_text(x))
labels = []
for index, row in df1.iterrows():
    if row['sentiment'] == 'neutral':
        row['sentiment'] = 0
    elif row['sentiment'] == 'positive':
        row['sentiment'] = 1
    elif row['sentiment'] == 'very_positive':
        row['sentiment'] = 1
    elif row['sentiment'] == 'negative':
        row['sentiment'] = 2
    else:
        row['sentiment'] = 2
    labels.append(row['sentiment'])
Y = np_utils.to_categorical(labels, 5)
vocabulary_size = 20000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df1['text'])
sequences = tokenizer.texts_to_sequences(df1['text'])
data = pad_sequences(sequences, maxlen=50)
# out = model_glove.evaluate(data)
results = model.evaluate(data, Y)
print(results)
