import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

df = pd.read_csv("../../data/averageCleaned.csv")

print(df.head().to_markdown(index=False, numalign="left", stralign="left"))

# Colonne categoriche
categorical_columns = ["DOC_TYPE", "PROC_TYPE", "DOSSIER_TYPE"]

# Separazione delle feature e del target
X = df.drop(["GROSS_SPA", "NET_SPA", "ID"], axis=1)
y = df[["GROSS_SPA", "NET_SPA"]]

# Applicazione della codifica one-hot
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)

# Converti la matrice sparsa in un array denso
X = X.toarray()

# Normalizzazione delle feature
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

# Normalizzazione dei target
scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y)

# Converti in tensori PyTorch
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dimensione degli input e output
input_size = X.shape[1]
output_size = y.shape[1]


# Definizione del modello
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.2)  # Dropout aggiunto qui

        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.2)  # Dropout aggiunto qui

        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.2)  # Dropout aggiunto qui

        self.fc4 = nn.Linear(hidden_size3, output_size)

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
        return out

# Iperparametri
hidden_size1 = 128
hidden_size2 = 64
hidden_size3 = 32
num_epochs = 500
learning_rate = 0.001

# Creazione del modello
model = NeuralNetwork(input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

# Spostamento del modello sulla GPU se disponibile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

from torch.utils.data import TensorDataset, DataLoader

# Creazione dei dataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64

# Creazione dei DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Training del modello
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        # Spostamento dei dati sulla GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Azzeramento dei gradienti
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass e ottimizzazione
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Stampa della perdita media per epoca
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.6f}')

# Valutazione del modello
model.eval()
with torch.no_grad():
    y_pred = []
    y_true = []
    for X_batch, y_batch in test_loader:
        # Spostamento dei dati sulla GPU
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs = model(X_batch)
        y_pred.append(outputs.cpu().numpy())
        y_true.append(y_batch.cpu().numpy())

    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)

# Inversa la normalizzazione dei target
y_test_inv = scaler_y.inverse_transform(y_true)
y_pred_inv = scaler_y.inverse_transform(y_pred)

# Calcolo del MSE
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'Test MSE: {mse}')

# Grafico dei valori previsti vs reali
plt.figure(figsize=(10, 6))
plt.scatter(y_test_inv[:, 0], y_pred_inv[:, 0], label='GROSS_SPA', alpha=0.6)
plt.scatter(y_test_inv[:, 1], y_pred_inv[:, 1], label='NET_SPA', alpha=0.6)
plt.xlabel('Valori Reali')
plt.ylabel('Valori Previsti')
plt.legend()
plt.title('Valori Previsti vs Valori Reali')
plt.savefig('Valori Previsti vs Valori Reali')

# Grafico dei residui
residui_gross = y_test_inv[:, 0] - y_pred_inv[:, 0]
residui_net = y_test_inv[:, 1] - y_pred_inv[:, 1]

plt.figure(figsize=(10, 6))
plt.hist(residui_gross, bins=50, alpha=0.6, label='GROSS_SPA Residui')
plt.hist(residui_net, bins=50, alpha=0.6, label='NET_SPA Residui')
plt.xlabel('Residui')
plt.ylabel('Frequenza')
plt.legend()
plt.title('Distribuzione dei Residui')
plt.savefig('Distribuzione dei Residui.png')

