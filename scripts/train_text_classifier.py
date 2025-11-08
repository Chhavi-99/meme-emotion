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
from keras.layers import Flatten
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


df = pd.read_csv('train_text_data.csv', names=[
                 'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df.describe())
df['text'] = df['text'].map(lambda x: clean_text(x))
labels = []
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

# print(labels)
# tokenization
vocabulary_size = 40000
tokenizer = Tokenizer(num_words=vocabulary_size)
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=100)

# model_glove = Conv1D(64, 5, activation='relu')(model_glove)
# model_glove = MaxPooling1D(pool_size=4)(model_glove)curacy'])

# model_lstm.fit(data, np.array(labels), validation_split=0.1, epochs=10)
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    #print(word, coefs)
    embeddings_index[word] = coefs
f.close()
# print(embeddings_index)
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocabulary_size, 100))
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:
        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

input1 = Input(shape=(100,))
model_glove = Embedding(vocabulary_size, 100, input_length=100,
                        weights=[embedding_matrix])(input1)
#model_glove = Dropout(0.2)(model_glove)
model_glove = Conv1D(64, 5, activation='relu')(model_glove)
model_glove = MaxPooling1D(pool_size=4)(model_glove)
#model_glove = Conv1D(64, 5, activation='relu')(model_glove)
#model_glove = Flatten()(model_glove)
model_glove = LSTM(100)(model_glove)
model_glove = Dense(4, activation='sigmoid')(model_glove)
model = Model(inputs=input1, outputs=model_glove)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(data, np.array(labels), validation_split=0.1, epochs=10, verbose=2)

df1 = pd.read_csv('text_test.csv', names=[
    'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text'])
# print(df1.describe())
df1['text'] = df1['text'].map(lambda x: clean_text(x))
labels = []
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
    labels.append([row['funny_image'], row['sarcastic_image'],
                   row['offensive_image'], row['motivational_image']])

sequences = tokenizer.texts_to_sequences(df1['text'])
data = pad_sequences(sequences, maxlen=100)
results = model.predict(data, np.array(labels))
print(results)
file=open('abc.csv','a')
for result in results:
    f = ''
    g = ''
    h = ''
    i = ''
    if result[0] >= 0.5:
        f='funny'
    else:
        f='not_funny'
    if result[1]>=0.5:
        g='sarcastic'
    else:
        g='not_sarcastic'
    if result[2]>=0.5:
        h='offensive'
    else:
        h='not_offensive'
    if result[3]>=0.5:
        i='motivational'
    else:
        i='not_motivational'
    file.write(f+','+g+','+h+','+i)
    file.write('\n')
file.close()
        
        

        # file = open('text_output.csv1', 'a')
        # for i in out:
        #     file.write(str(i[0])+','+str(i[1])+','+str(i[2])+','+str(i[3]))
        #     file.write('\n')
        # file.close()
        # file = open('text_output.csv2', 'a')
        # for index, row in df1.iterrows():
        #     file.write(row['image_name'])
        #     file.write('\n')
        # file.close()
