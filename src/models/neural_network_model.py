import torch.nn as nn

class ComplexNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        super(ComplexNeuralNetwork, self).__init__()
        # Primo livello
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)

        # Secondo livello
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)

        # Terzo livello
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.3)

        # Quarto livello
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.3)

        # Output layer
        self.fc5 = nn.Linear(hidden_size4, output_size)
        self.activation_out = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.dropout3(out)

        out = self.fc4(out)
        out = self.relu4(out)
        out = self.dropout4(out)

        out = self.fc5(out)
        return self.activation_out(out)
