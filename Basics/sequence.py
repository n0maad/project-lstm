'''
PyTorch expects all LSTM inputs to be 3D Tensors.
The scehmatic of the axes of these tensors are important:
    First axis is the sequence itself
    Second indexes instances in the mini-batch  (for this example we will use a 1D for the second axis)
    Third axis indexes the elements of the input

It will not be used in this exercise but research Viterbi and determine how it could be used within this project.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3,3)                             # input and output dimensions iare 3
inputs  = [torch.randn(1,3) for _ in range(5)]  # make a sequence of length 5

# Initialise the hidden states
hidden = (torch.randn(1,1,3), torch.randn(1,1,3))

for i in inputs:
    # step through the sequence one element at a time
    # after each step, hidden contains the hidden state
    out, hidden = lstm(i.view(1,1,-1), hidden)  

'''
    Alternatively, the entire sequence can be done at once.
    The first value returned by the LSTM is all of the hidden states throughout the sequence.
    The second is just the most recent hidden state (compare the last slice of "out" with "hidden" below, they are the same)

    The reason for that is because "out" will give access to all hidden states in the sequence,
    whilst "hidden" will allow the network to continue the sequence and backpropagate, by passing it as an argument to the LSTM (at a later date).
'''

# Add the second dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
hidden = (torch.randn(1,1,3), torch.randn(1,1,3))   # clean out hidden state
out, hidden = lstm(inputs, hidden)

print(out)
print(hidden)


# Prepare data
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

training_data = [('The dog ate the apple'.split(),['DET', 'NN', 'V', 'DET','NN']),
                 ('Everybody read that book'.split(),['NN', 'V', 'DET', 'NN'])]

word_to_ix = {} # creates an empty dictionary

for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

tag_to_ix = {'DET':0, 'NN':1, 'V':2} #keys:values - a collection that is unordered, changeable and indexed

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# Create the model
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        # the LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # the linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embedding(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim = 1)
        return tag_scores

# Training the model

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimiser = optim.SGD(model.parameters(), lr=0.1)

# See what scoares are before training
# Note that element i, j of the output is the score for tag j for word i
# No training required as code is wrapped in torch.no_grad() - Determine what this means.
with torch.no_grad():       
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)


for epoch in range(300):
    for sentence, tags in training_data:
        # Step 1 - Pytorch accumulates gradients, so they must be cleared out for each instance
        model.zero_grad()   # clears gradients
        
        # Step 2 - Convert inputs into Tensors of word indices
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3 - Runs a forward pass
        tag_scores = model(sentence_in)

        # Step 4 - Compute the loss, gradients and update the parameters
        loss = loss_function(tag_scores, targets)   # computes the loss
        loss.backward()                             # computes the gradients
        optimiser.step()                            # updates the parameters

# See scores after training
with torch.no_grad():       
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)

'''
    The sentence is "the dog ate the apple".
    i, j correspond to the score for tag j
    The smaller the value dictates the networks assumption of what the correct answer is
'''






