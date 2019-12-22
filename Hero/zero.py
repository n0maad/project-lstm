import torch
import torch.nn as nn
import bz2
import re
import nltk
import numpy as np

nltk.download('punkt')

from collections import Counter
from torch.utils.data import TensorDataset, DataLoader



train_file  = bz2.BZ2File('../amazon_reviews/train.ft.txt.bz2')
test_file   = bz2.BZ2File('../amazon_reviews/test.ft.txt.bz2')

train_file  = train_file.readlines()
test_file   = test_file.readlines()

print("Numbers of training reviews: " + str(len(train_file)))
print("Numbers of test reviews: " + str(len(test_file)))

num_train   = 800000    # We're training on the first 800,000 reviews in the dataset
num_test    = 200000    # Using 200,000 reviers for the test

train_file  = [x.decode('utf-8') for x in train_file[:num_train]]
test_file   = [x.decode('utf-8') for x in test_file[:num_test]]

#print(train_file[0])

# Extracting labesl from sentences
train_labels    = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file] # this split the sentences into individual works and replaces initial space with __label__1
train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]             # splits the sentence into two sections and removes the last character from the sentence

test_labels    = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]   # this split the sentences into individual works and replaces initial space with __label__1
test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]               # splits the sentence into two sections and removes the last character from the sentence

# Cleaning data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d', '0', train_sentences[i])  # replace any digit(0-9) with the value of 0

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d', '0', test_sentences[i])  # replace any digit(0-9) with the value of 0

# Modifying the URL link
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i]) # ([^ ]+(?<=\.[a-z]{3})) is known as a digital expression (digex)

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()                               # Dictonary that will map a word to the number of times it appeared in all the training sentences
for i, sentence in enumerate(train_sentences):
    # the sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence):   # tokenising the words by seperating each word in the sentence into its own token.
        words.update([word.lower()])            # converts all words to lowercase
        train_sentences[i].append(word)
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
for i, sentences in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence] # 1(url) or 0(git)?

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokensize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences will be padded/shortened to
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len), dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 200

train_sentences = pad_input(train_sentences, seq_len)
test_sentences  = pad_input(test_sentences, seq_len)

# Converting labels into numpy arrays
train_labels    = np.array(train_labels)
test_labels     = np.array(test_labels)

# Splitting data into testing and validating sets
split_frac  = 0.5                                   # splitting data 50/50
split_id    = int(split_frac * len(sentences))      # determining half the value of all reviews

val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

# Using PyTorch DataLoader to define the datasets
train_data  = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data    = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data   = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

batch_size  = 400

train_loader    = DataLoader(train_data, shuffle = True, batch_size = batch_size)
val_loader      = DataLoader(val_data, shuffle = True, batch_size = batch_size)
test_loader     = DataLoader(test_data, shuffle = True, batch_size = batch_size)

# Set GPU to process data
is_cuda = torch.cuda.is_available()

if is_cuda:
    device  = torch.device('cuda')
    print('GPU is available')
else:
    device  = torch.device('cpu')
    print('CPU used, GPU not available')

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob = 0.5):
        super(SentimentNet, self).__init__()
        self.output_size    = output_size
        self.n_layers       = n_layers
        self.hidden_dim     = hidden_dim

        self.embedding      = nn.Embedding(vocab_size, embedding_dim)
        self.lstm           = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout = drop_prob, batch_first = True)
        self.dropout        = nn.Dropout(drop_prob)
        self.fc             = nn.Linear(hidden_dim, output_size)
        self.sigmoid        = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size  = x.size(0)
        x = x.long()
        embeds  = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.sigmoid(out)

        out = out.view(batch_size, -1)
        out = out[:,-1]
        return out, hidden
    
    def init_hidden(self, batch_size):
        weight  = next(self.parameters()).data
        hidden  = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device), weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))
        return hidden

vocab_size  = len(word2idx) + 1
output_size = 1
embedding_dim   = 400
hidden_dim  = 512
n_layers    = 2

model  = SentimentNet(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
model.to(device)
print(model)

lr  = 0.005
criterion   = nn.BCELoss()
optimiser   = torch.optim.Adam(model.parameters(), lr = lr)

epochs  = 2
counter = 0
print_every = 1000
clip    = 5
valid_loss_min  = np.Inf

model.train()
for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output, h = model(inputs, h)
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimiser.step()

        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out, val_h = model(inp, val_h)
                val_loss = criterion(out.squeeze(), lab.float())
                val_losses.append(val_loss.item())
            
            model.train()

            print('Epoch: {}/{}...'.format(i+1, epochs), "Steps: {}...".format(counter), "Loss: {:.6f}".format(loss.item()), "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) <= valid_loss_min:
                torch.save(model.state_dict(), './state_dict.pt')
                print('Validation loss decreased ({:.6f}-->{:.6f}).    Saving model...'.format(valid_loss_min, np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# loading the best model
model.load_state_dict(torch.load('./state_dict.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()
for inputs, labels in test_loader:
    h   = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    #model.zero_grad()
    output, h = model(inputs, h)
    test_loss = criterion(out.squeeze(), labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())    # rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

print("Test loss: {:.3f}".format(np.mean(test_loss)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
