# Import library to be used
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional, TimeDistributed, Concatenate
from keras.preprocessing.text import Tokenizer

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

# data = [3, 7, 1, 15, 7]

# s = ku.to_categorical(data, num_classes=100)
# print(s)
# sys.exit()


def generate_padded_sequences(input_sequences, total_words):
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='post'))

    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    # print(input_sequences[:, 1])
    # print(len(input_sequences[:, :-1]))
    label = ku.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len


def get_dataset(story_seq, summary_seq, total_features):
    X1, X2, y = list(), list(), list()

    for i, (story, summary) in enumerate(zip(story_seq, summary_seq)):
        # print(" >>> : ", i),
        # print(len(story))
        print(summary),


# # load the review dataset from pickle
loaded_stories = load(open("review_dataset.pkl", "rb"))


# print(loaded_stories[0]['story'])

# print(type(loaded_stories))

df = pd.DataFrame(loaded_stories)


# df = df[:1000]

dataset_length = len(df)
train_len = 0.80
test_len = 0.20


story_list = list(df['story'].values)
summary_list = list(df['highlights'].values)


tokenizer_story = Tokenizer()
tokenizer_summary = Tokenizer()

tokenizer_story.fit_on_texts(story_list)
tokenizer_summary.fit_on_texts(summary_list)


listed_story_sequence = tokenizer_story.texts_to_sequences(story_list)
listed_summary_sequence = tokenizer_summary.texts_to_sequences(summary_list)


max_sequence_story_len = max([len(story) for story in listed_story_sequence])
max_sequence_summary_len = max([len(summary)
                                for summary in listed_summary_sequence])


# print(max_sequence_story_len, " -- ", max_sequence_summary_len)
# sys.exit(0)

story_seq = pad_sequences(listed_story_sequence,
                          max_sequence_story_len, padding="post")

summary_seq = pad_sequences(listed_summary_sequence,
                            max_sequence_summary_len, padding="post")


d = summary_seq[:-1]

arr = np.zeros(len(summary_seq[0]))
decoded_data = np.insert(d, 0, arr, axis=0)
# print(summary_seq[0])

X1, X2, y = list(), list(), list()
for (story, summary) in zip(story_seq, summary_seq):
    source = story
    target = summary
    # target_in = target[:-1]
    target_in = target[:-1]
    target_in = np.insert(target_in, 0, 0, axis=0)

    # src_encoded = ku.to_categorical(
    #     [source], num_classes=len(tokenizer_story.word_index) + 1)[0]
    tar_encoded = ku.to_categorical(
        [target], num_classes=len(tokenizer_story.word_index) + 1)[0]
    # tar2_encoded = ku.to_categorical(
    #     [target_in], num_classes=len(tokenizer_story.word_index) + 1)[0]

    # X1.append(src_encoded)
    # X2.append(tar2_encoded)
    # y.append(tar_encoded)

    X1.append(source)
    X2.append(target_in)
    y.append(tar_encoded)


X1 = np.array(X1)
X2 = np.array(X2)
y = np.array(y)


# label = summary_seq[:, :-1], summary_seq[:, -1]

# X_story_train = story_seq[: int(dataset_length * train_len)]
# y_story_train = summary_seq[: int(dataset_length * train_len)]
# X_story_test = story_seq[: -int(dataset_length * train_len)]
# y_sum_test = summary_seq[: -int(dataset_length * train_len)]


vocabulary_size_story = len(tokenizer_story.word_index) + 1
vocabulary_size_summary = len(tokenizer_summary.word_index) + 1

outputBatch = 8
# inputLength = 20


encoder_inputs = Input(shape=(None, ))
enc_emb = Embedding(vocabulary_size_story, outputBatch)(encoder_inputs)
# encoder = LSTM(1000, return_state=True)
# encoder_output, state_h, state_c = encoder(enc_emb)

# We discard `encoder_outputs` and only keep the states.
# encoder_states = [state_h, state_c]

encoder = Bidirectional(LSTM(1000, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(
    enc_emb)

state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])

encoder_states = [state_h, state_c]


# Decoder Model
# setup decoder using `encoder_state` as initial state
decoder_inputs = Input(shape=(None, ))

dec_emb = Embedding(vocabulary_size_summary, outputBatch)

final_dec_embedding = dec_emb(decoder_inputs)
decoder_lstm = LSTM(1000 * 2, return_sequences=True, return_state=True)

decoder_outputs, _, _ = decoder_lstm(
    final_dec_embedding, initial_state=encoder_states)

# decoder_dense = TimeDistributed(
#     Dense(vocabulary_size_story, activation='softmax'))

decoder_dense = Dense(vocabulary_size_story, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])

# sys.exit(0)
model.summary()
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([X1, X2], y,
          batch_size=10, epochs=1, validation_split=0.3)


sys.exit(0)
reds = model.predict_classes(X_story_test)


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
