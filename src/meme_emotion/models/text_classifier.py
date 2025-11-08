from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import string
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import matplotlib as plt
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.callbacks import *
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import confusion_matrix


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


df1 = pd.read_csv('text_train.csv', names=[
    'image_name', 'funny_image', 'sarcastic_image', 'offensive_image', 'motivational_image', 'text', 'val'])
columns = ['image_name', 'funny_image', 'sarcastic_image',
           'offensive_image', 'motivational_image']
df1.drop(columns, inplace=True, axis=1)
print(df1.shape)

df2 = pd.read_csv('Flickr_7kdata.csv', names=[
                  'image_name', 'image_id', 'text', 'val'], sep='|')
columns = ['image_name', 'image_id']
df2.drop(columns, inplace=True, axis=1)
print(df2.shape)

frames = [df1, df2]
df = pd.concat(frames)
# df.sample(frac=1)
print(df.shape)

df['text'] = df['text'].map(lambda x: clean_text(x))
#df = shuffle(df, random_state=42)

# print(df.head(20))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['text'])
vocabulary_size = len(tokenizer.word_index)+1
max_length = max([len(s.split()) for s in df['text']])

sequences = tokenizer.texts_to_sequences(df['text'])
data = pad_sequences(sequences, maxlen=max_length)

X_train, X_test, y_train, y_test = train_test_split(
    data, np.array(df['val']), test_size=0.1, random_state=42)


# print(tokenizer.word_index)


embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((vocabulary_size, 100))
print(embedding_matrix.shape)
for word, index in tokenizer.word_index.items():
    if index > vocabulary_size - 1:

        break
    else:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

model_glove = Sequential()
model_glove.add(Embedding(vocabulary_size, 100, input_length=max_length,
                          weights=[embedding_matrix]))
model_glove.add(LSTM(100))
model_glove.add(Dense(1, activation='sigmoid'))
model_glove.compile(loss='binary_crossentropy',
                    optimizer='adam', metrics=['accuracy'])
model_glove.summary()
es = EarlyStopping(monitor='val_accuracy', mode='auto',
                   restore_best_weights=True, verbose=1, patience=6)
model_glove.fit(X_train, y_train, validation_split=0.1,
                callbacks=[es], epochs=10)

yhat_probs = model_glove.predict(X_test, verbose=0)
yhat_classes = model_glove.predict_classes(X_test, verbose=0)
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
