import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)  # Tokenize the pattern (split into words)
        all_words.extend(w)  # Add the words to the list of all words
        xy.append((w, tag))  # Add the (word list, tag) pair to xy

# Example content of 'xy':
# [('Hi', 'greeting'), ('Hey', 'greeting'), ('How', 'greeting'), ...]

# Example content of 'all_words':
# ['Hi', 'Hey', 'How', 'are', 'you', 'Is', 'anyone', 'there', ...]

ignore_words = ['?', '!', '.', ',']
# Stemming of words and removal of ignored words
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Sorting and removing duplicates to get a list of unique words
all_words = sorted(set(all_words))
# Sorting and removing duplicates to get a list of unique tags
tags = sorted(set(tags))

# Initialize lists for training data
X_train = []  # Model inputs (word vectors)
Y_train = []  # Model outputs (intent labels)

# Construct training data sets
for (pattern_sentence, tag) in xy:
    # Create the 'bag of words' for the sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)  # Add the BoW vector to X_train

    # Get the index of the tag in the list of tags
    label = tags.index(tag)
    Y_train.append(label)  # Add the tag index to Y_train

# Convert lists to NumPy arrays for training
X_train = np.array(X_train)
Y_train = np.array(Y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    #dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


# Parameters
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

# Save the data
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = 'chat_model.pth'
torch.save(data, FILE)
print(f'Saved model to {FILE}')
