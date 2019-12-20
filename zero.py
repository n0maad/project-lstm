import torch
import torch.nn as nn
import bz2
from collections import Counter
import re
import nltk
import numpy as np
nltk.download('punkt')


train_file  = bz2.BZ2File('./amazon_reviews/train.ft.txt.bz2')
test_file   = bz2.BZ2File('./amazon_reviews/test.ft.txt.bz2')

train_file  = train_file.readlines()
test_file   = test_file.readlines()

print("Numbers of training reviews: " + str(len(train_file)))
print("Numbers of test reviews: " + str(len(test_file)))

num_train   = 800000    # We're training on the first 800,000 reviews in the dataset
num_test    = 200000    # Using 200,000 reviers for the test

train_file  = [x.decode('utf-8') for x in train_file[:num_train]]
test_file   = [x.decode('utf-8') for x in test_file[:num_test]]

#print(train_file[0])

# Extracting labesl from sentances
train_labels    = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file] # this split the sentances into individual works and replaces initial space with __label__1
train_sentances = [x.split(' ', 1)[1][:-1].lower() for x in train_file]             # splits the sentance into two sections and removes the last character from the sentance

test_labels    = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]   # this split the sentances into individual works and replaces initial space with __label__1
test_sentances = [x.split(' ', 1)[1][:-1].lower() for x in test_file]               # splits the sentance into two sections and removes the last character from the sentance

# Cleaning data
for i in range(len(train_sentances)):
    train_sentances[i] = re.sub('\d', '0', train_sentances[i])  # replace any digit(0-9) with the value of 0

for i in range(len(test_sentances)):
    test_sentances[i] = re.sub('\d', '0', test_sentances[i])  # replace any digit(0-9) with the value of 0

# Modifying the URL link
for i in range(len(train_sentances)):
    if 'www.' in train_sentances[i] or 'http:' in train_sentances[i] or 'https:' in train_sentances[i]:
        train_sentances[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentances[i]) # ([^ ]+(?<=\.[a-z]{3})) is known as a digital expression (digex)

for i in range(len(test_sentances)):
    if 'www.' in test_sentances[i] or 'http:' in test_sentances[i] or 'https:' in test_sentances[i]:
        test_sentances[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentances[i])

words = Counter()                               # Dictonary that will map a word to the number of times it appeared in all the training sentances
for i, sentance in enumerate(train_sentances):
    # the sentances will be stored as a list of words/tokens
    train_sentances[i] = []
    for word in nltk.word_tokenize(sentance):   # tokenising the words by seperating each word in the sentance into its own token.
        words.update([word.lower()])            # converts all words to lowercase
        train_sentances[i].append(word)
    if i%20000 == 0:
        print(str((i*100)/num_train) + "% done")
print("100% done")

# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}

# Sorting the words according to the number of appearances, with the most common word being first.
words = sorted(words, key=words.get, reverse=True)

# Adding padding and inknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

# Convert words from sentences to their corresponding indexes
for i, sentences in enumerate(train_sentances):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence]

for i, sentence in enumerate(test_sentances):
    # For test sentences, we have to tokensize the sentences as well
    test_sentances[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]


# Defining a function that either shortens sentences or pads sentences will be padded/shortened to