import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # First linear layer with 'input_size' inputs and 'hidden_size' outputs
        self.l1 = nn.Linear(input_size, hidden_size)
        # Second linear layer with 'hidden_size' inputs and 'hidden_size' outputs
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Third linear layer with 'hidden_size' inputs and 'num_classes' outputs
        self.l3 = nn.Linear(hidden_size, num_classes)
        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input 'x' through the first linear layer and then ReLU
        out = self.l1(x)
        out = self.relu(out)
        # Pass the result through the second linear layer and then ReLU
        out = self.l2(out)
        out = self.relu(out)
        # Pass the result through the third linear layer
        out = self.l3(out)
        # No activation or softmax at the end
        return out
