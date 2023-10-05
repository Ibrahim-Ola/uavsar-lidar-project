import torch.nn as nn

class RegressionNN(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3):
        super(RegressionNN, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(input_size, hidden_size1)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)

        # third fully connected layer
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        
        # Output layer
        self.fc4 = nn.Linear(hidden_size3, 1)
        
        # ReLU activation
        self.relu = nn.ReLU()
        
        # Dropout layer 1
        #self.dropout1 = nn.Dropout(0.1)

        # Dropout layer 2
        #self.dropout2 = nn.Dropout(0.15)

    def forward(self, x):
        x = self.relu(self.fc1(x))     # First hidden layer with ReLU activation
        #x = self.dropout1(x)            # Dropout after first hidden layer
        
        x = self.relu(self.fc2(x))     # Second hidden layer with ReLU activation
        #x = self.dropout2(x)            # Dropout after second hidden layer
        
        x = self.relu(self.fc3(x))     # Third hidden layer with ReLU activation

        x = self.fc4(x)                # Output layer (no activation here for regression)
        return x
