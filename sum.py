# Import library to be used
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Bidirectional
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

p, label, m = generate_padded_sequences(
    listed_summary_sequence, len(tokenizer.word_index) + 1)

# print(len(label[0]))
# sys.exit(0)

# random.shuffle(story_seq)
# random.shuffle(summary_seq)


# X_story_train = story_seq[: int(dataset_length * train_len)]
# y_story_train = summary_seq[: int(dataset_length * train_len)]
# X_sum_test = story_seq[: -int(dataset_length * train_len)]
# y_sum_test = summary_seq[: -int(dataset_length * train_len)]

# label

# model = Sequential()

vocabulary_size = len(tokenizer.word_index) + 1
outputBatch = 8
# inputLength = 20

# Define an input sequence and process it.
encoder_inputs = Input(shape=(max_sequence_story_len, ))
x = Embedding(vocabulary_size, outputBatch)(encoder_inputs)
x, state_h, state_c, = LSTM(outputBatch, return_state=True)(x)
encoder_states = [state_h, state_c]

# setup decoder using `encoder_state` as initial state
decoder_inputs = Input(shape=(max_sequence_summary_len,))
x = Embedding(vocabulary_size, outputBatch)(decoder_inputs)
x = LSTM(outputBatch)(x, initial_state=encoder_states)
# decoder_lstm, _, _ = x(decoder_inputs, initial_state=encoder_states)
# x = LSTM(outputBatch, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(vocabulary_size, activation='softmax')(x)


# Define the model that will turn
# `encoder_input_data` & `decoder =_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# compile & run training
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!
model.fit([story_seq, summary_seq], label,
          batch_size=10, epochs=10, validation_split=0.2)


# model.add(Embedding(vocabulary_size, outputBatch, input_length=max_sequence_len))
# model.add(Bidirectional(LSTM(300, return_sequences=True)))
# model.add(Bidirectional(LSTM(100)))
# model.add(Dropout(0.2))
# model.add(Dense(max_sequence_len, activation='softmax'))
# # model.compile(loss="categorical_crossentropy", optimizer="adam")
# model.compile(loss="mse", optimizer="rmsprop")
# model.fit(x=X_story_train, y=y_story_train, steps_per_epoch=5, epochs=10,
#           validation_data=(X_story_test, y_story_test), validation_steps=2)
# # model.summary()


sys.exit(0)

# Inference
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(outputBatch,))
decoder_state_input_c = Input(shape=(outputBatch,))

decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)

decoder_states = [state_h, state_c]

decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
                len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence
