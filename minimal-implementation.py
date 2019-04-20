# Import library to be used
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import RepeatVector, Input, Dense, LSTM, Embedding, Dropout, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras import optimizers

import random

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping
import keras.utils as ku


import pandas as pd
import numpy as np
import string
import os
import sys

from pickle import load, dump


def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word

    return None


def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='post'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    # print(input_sequences[:, 1])
    # print(len(input_sequences[:, :-1]))
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


# # load the review dataset from pickle
loaded_stories = load(open("review_dataset.pkl", "rb"))


# print(loaded_stories[0]['story'])

# print(type(loaded_stories))

df = pd.DataFrame(loaded_stories)


df = df[:100]

dataset_length = len(df)
train_len = 0.80
test_len = 0.20


story_list = list(df['story'].values)
summary_list = list(df['highlights'].values)


tokenizer = Tokenizer()

tokenizer.fit_on_texts(story_list)

listed_story_sequence = tokenizer.texts_to_sequences(story_list)
listed_summary_sequence = tokenizer.texts_to_sequences(summary_list)


# print(tokenizer.word_index.items())

# for word, index in tokenizer.word_index.items():
#     print(index, " -- ", word)

# w = get_word(1382, tokenizer)
# print(w)
# sys.exit(0)

max_sequence_story_len = max([len(story) for story in listed_story_sequence])
max_sequence_summary_len = max([len(summary)
                                for summary in listed_summary_sequence])


# print(max_sequence_story_len, " -- ", max_sequence_summary_len)
# sys.exit(0)

story_seq = pad_sequences(listed_story_sequence,
                          max_sequence_story_len, padding="post")

summary_seq = pad_sequences(listed_summary_sequence,
                            max_sequence_summary_len, padding="post")


######

# label = summary_seq[:, :-1], summary_seq[:, -1]
# label = ku.to_categorical(
#     label, num_classes=len(tokenizer.word_index) + 1)

# p, label, m = generate_padded_sequences(
#     listed_summary_sequence, len(tokenizer.word_index) + 1)

# print(len(label[0]))
# sys.exit(0)

random.shuffle(story_seq)
random.shuffle(summary_seq)


X_story_train = story_seq[: int(dataset_length * train_len)]
y_story_train = summary_seq[: int(dataset_length * train_len)]
X_sum_test = story_seq[: -int(dataset_length * train_len)]
y_sum_test = summary_seq[: -int(dataset_length * train_len)]

# label
# model = Sequential()

rms = optimizers.RMSprop(lr=0.001)

vocabulary_size = len(tokenizer.word_index) + 1
outputBatch = 8
# inputLength = 20

model = Sequential()
model.add(Embedding(vocabulary_size, outputBatch,
                    input_length=max_sequence_story_len))
model.add(LSTM(1000))
model.add(RepeatVector(max_sequence_summary_len))
model.add(LSTM(1000, return_sequences=True))
model.add(Dense(vocabulary_size, activation='softmax'))
model.compile(
    optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['acc'])

history = model.fit(X_story_train,   y_story_train.reshape(
    y_story_train.shape[0], y_story_train.shape[1], 1), epochs=500, batch_size=64, validation_split=0.2)


preds = model.predict_classes(X_sum_test.reshape(
    (X_sum_test.shape[0], X_sum_test.shape[1])))


# print(preds)

preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], tokenizer)
        if j > 0:
            if (t == get_word(i[j-1], tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t)

    preds_text.append(' '.join(temp))


print(preds_text)
