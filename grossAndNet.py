import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

df = pd.read_csv("resources/averageCleaned.csv")

categorical_columns = ["DOC_TYPE", "PROC_TYPE", "DOSSIER_TYPE"]

# Separazione delle feature e dei target
X = df.drop(["GROSS_SPA", "NET_SPA", "ID"], axis=1)
y_gross = df["GROSS_SPA"]
y_net = df["NET_SPA"]

# Applicazione della codifica one-hot per le feature categoriali
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)
X = X.toarray()

# Normalizzazione delle feature
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

# Normalizzazione dei target GROSS_SPA e NET_SPA separatamente
scaler_y_gross = MinMaxScaler()
y_gross = scaler_y_gross.fit_transform(y_gross.values.reshape(-1, 1))

scaler_y_net = MinMaxScaler()
y_net = scaler_y_net.fit_transform(y_net.values.reshape(-1, 1))

# Converti in tensori PyTorch
X = torch.tensor(X, dtype=torch.float32)
y_gross = torch.tensor(y_gross, dtype=torch.float32)
y_net = torch.tensor(y_net, dtype=torch.float32)

# Suddivisione dei dati in training e test set
X_train, X_test, y_gross_train, y_gross_test = train_test_split(X, y_gross, test_size=0.2, random_state=42)
X_train, X_test, y_net_train, y_net_test = train_test_split(X, y_net, test_size=0.2, random_state=42)

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Iperparametri
hidden_size1 = 128
hidden_size2 = 64
num_epochs = 500
learning_rate = 0.001

# Modello per GROSS_SPA
model_gross = NeuralNetwork(X.shape[1], hidden_size1, hidden_size2, 1).to(device)
criterion = nn.MSELoss()
optimizer_gross = optim.Adam(model_gross.parameters(), lr=learning_rate)

# Modello per NET_SPA
model_net = NeuralNetwork(X.shape[1], hidden_size1, hidden_size2, 1).to(device)
optimizer_net = optim.Adam(model_net.parameters(), lr=learning_rate)

# Addestramento per GROSS_SPA
train_dataset_gross = TensorDataset(X_train, y_gross_train)
train_loader_gross = DataLoader(dataset=train_dataset_gross, batch_size=64, shuffle=True)

model_gross.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader_gross:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_gross.zero_grad()
        outputs = model_gross(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_gross.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss for GROSS_SPA: {epoch_loss / len(train_loader_gross):.6f}')

# Addestramento per NET_SPA
train_dataset_net = TensorDataset(X_train, y_net_train)
train_loader_net = DataLoader(dataset=train_dataset_net, batch_size=64, shuffle=True)

model_net.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader_net:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer_net.zero_grad()
        outputs = model_net(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer_net.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss for NET_SPA: {epoch_loss / len(train_loader_net):.6f}')

# Valutazione per GROSS_SPA
model_gross.eval()
with torch.no_grad():
    y_gross_pred = model_gross(X_test.to(device)).cpu().numpy()
    y_gross_test_inv = scaler_y_gross.inverse_transform(y_gross_test.cpu().numpy())
    y_gross_pred_inv = scaler_y_gross.inverse_transform(y_gross_pred)

# Calcolo del MSE per GROSS_SPA
mse_gross = mean_squared_error(y_gross_test_inv, y_gross_pred_inv)
print(f'Test MSE for GROSS_SPA: {mse_gross}')

# Valutazione per NET_SPA
model_net.eval()
with torch.no_grad():
    y_net_pred = model_net(X_test.to(device)).cpu().numpy()
    y_net_test_inv = scaler_y_net.inverse_transform(y_net_test.cpu().numpy())
    y_net_pred_inv = scaler_y_net.inverse_transform(y_net_pred)

# Calcolo del MSE per NET_SPA
mse_net = mean_squared_error(y_net_test_inv, y_net_pred_inv)
print(f'Test MSE for NET_SPA: {mse_net}')
