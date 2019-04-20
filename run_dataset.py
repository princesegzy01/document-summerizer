import pandas as pd
import re
import os
import sys

from nltk.corpus import stopwords
from pickle import load, dump

reviews = pd.read_csv("Reviews.csv")
reviews = reviews.dropna()

reviews = reviews.drop(['Id', 'ProductId', 'UserId', 'ProfileName',
                        'HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time'], axis=1)

reviews = reviews.reset_index(drop=True)


# reviews = reviews[0: 5000]

# for i in range(5):

#     print("Review #", i+1)

#     print(reviews.Summary[i])

#     print(reviews.Text[i])

#     print()

# print(reviews.head())

contractions = {

    "ain't": "am not",

    "aren't": "are not",

    "can't": "cannot",

    "can't've": "cannot have",

    "'cause": "because",

    "could've": "could have",

    "couldn't": "could not",

    "couldn't've": "could not have",

    "didn't": "did not",

    "doesn't": "does not",

    "don't": "do not",

    "hadn't": "had not",

    "hadn't've": "had not have",

    "hasn't": "has not",

    "haven't": "have not",

    "he'd": "he would",

    "he'd've": "he would have"
}


def clean_text(text, remove_stopwords=True):

    # convert word to lower case
    text = text.lower()

    if True:
        text = text.split()

        new_text = []

        for word in text:
            if word in contractions:
                new_text.append(contractions[word])

            else:
                new_text.append(word)

            text = " ".join(new_text)
            text = re.sub(r'https?:\/\/.*[\r\n]*',
                          '', text, flags=re.MULTILINE)

            text = re.sub(r'\<a href', ' ', text)

            text = re.sub(r'&amp;', '', text)

            text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)

            text = re.sub(r'<br />', ' ', text)

            text = re.sub(r'\'', ' ', text)

            if remove_stopwords:
                text = text.split()
                stops = set(stopwords.words("english"))

                text = [w for w in text if not w in stops]

                text = " ".join(text)

        return text


clean_summaries = []

# print(reviews["Text"].loc[4])
# print(reviews.Text[4])

# clean summary

for summary in reviews.Summary:
    summary = clean_text(summary, remove_stopwords=True)
    clean_summaries.append(summary)
    # print(summary)

print("Summaries are complete.")

# clean text

clean_texts = []

for text in reviews.Text:
    clean_texts.append(clean_text(text))

print("Text are complete")

stories = list()

for i, text in enumerate(clean_texts):
    stories.append({'story': text, 'highlights': clean_summaries[i]})

    # save to file

dump(stories, open('review_dataset.pkl', 'wb'))

print("Done pickling")


# print("starting Encoder - Decoder")


# # hyperparameters
# batch_size = 64

# epoch = 10

# latent_dim = 256

# num_sample = 10000


# # load the review dataset from pickle
# loaded_stories = load(open("review_dataset.pkl", "rb"))

# print(loaded_stories)
