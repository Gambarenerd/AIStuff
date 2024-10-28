import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
    print(f"Number of CUDA cores: {torch.cuda.get_device_properties(0).multi_processor_count}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

df = pd.read_csv("resources/averageCleaned.csv")

categorical_columns = ["DOC_TYPE", "PROC_TYPE", "DOSSIER_TYPE"]

# Winsorization
#df['GROSS_SPA'] = np.clip(df['GROSS_SPA'], None, 50)
#df['NET_SPA'] = np.clip(df['NET_SPA'], None, 50)

# Filter out outliers bigger than 40
df = df[(df['GROSS_SPA'] <= 550) & (df['NET_SPA'] <= 350)]

# Dividing feature
X = df.drop(["GROSS_SPA", "NET_SPA", "ID"], axis=1)
y_gross = df["GROSS_SPA"]
y_net = df["NET_SPA"]

# One-hot encoding dropping the first to avoid multicollinearity
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)
X = X.toarray()

# Normalization
scaler_X = RobustScaler()
X = scaler_X.fit_transform(X)

#Logaritmic transform to handle skewed data
y_gross = np.log1p(y_gross.values)  # log(1 + y) for values close to zero
y_net = np.log1p(y_net.values)

# Normalization
scaler_y_gross = MinMaxScaler()
y_gross = scaler_y_gross.fit_transform(y_gross.reshape(-1, 1))

# Bette keep normalization otherwise the MAE increase
scaler_y_net = MinMaxScaler()
y_net = scaler_y_net.fit_transform(y_net.reshape(-1, 1))

# PyTorch tensor
X = torch.tensor(X, dtype=torch.float32)
y_gross = torch.tensor(y_gross, dtype=torch.float32)
y_net = torch.tensor(y_net, dtype=torch.float32)

# Splitting data in training e test set
X_train, X_test, y_gross_train, y_gross_test, y_net_train, y_net_test = train_test_split(
    X, y_gross, y_net, test_size=0.2, random_state=42)

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
        out = self.activation_out(out)
        return out

num_epochs = 500
learning_rate = 0.0001
hidden_size1 = 512
hidden_size2 = 256
hidden_size3 = 128
hidden_size4 = 64

# Modello complesso per GROSS_SPA
model_gross = ComplexNeuralNetwork(X.shape[1], hidden_size1, hidden_size2, hidden_size3, hidden_size4, 1).to(device)
criterion = nn.L1Loss()
optimizer_gross = optim.Adam(model_gross.parameters(), lr=learning_rate, weight_decay=1e-4)

# Modello complesso per NET_SPA
model_net = ComplexNeuralNetwork(X.shape[1], hidden_size1, hidden_size2, hidden_size3, hidden_size4, 1).to(device)
optimizer_net = optim.Adam(model_net.parameters(), lr=learning_rate, weight_decay=1e-4)


# Training for GROSS_SPA
train_dataset_gross = TensorDataset(X_train, y_gross_train)
train_loader_gross = DataLoader(dataset=train_dataset_gross, batch_size=128, shuffle=True)

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

# Training for NET_SPA
train_dataset_net = TensorDataset(X_train, y_net_train)
train_loader_net = DataLoader(dataset=train_dataset_net, batch_size=512, shuffle=True)

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

# Valutation GROSS_SPA
model_gross.eval()
with torch.no_grad():
    y_gross_pred = model_gross(X_test.to(device)).cpu().numpy()
    y_gross_test_inv = scaler_y_gross.inverse_transform(y_gross_test.cpu().numpy())
    y_gross_pred_inv = scaler_y_gross.inverse_transform(y_gross_pred)

    # Inverse transformation
    y_gross_test_inv = np.expm1(y_gross_test_inv)  # inversa di log(1 + y)
    y_gross_pred_inv = np.expm1(y_gross_pred_inv)

# MSE and MAE for GROSS_SPA
mse_gross = mean_squared_error(y_gross_test_inv, y_gross_pred_inv)
mae_gross = mean_absolute_error(y_gross_test_inv, y_gross_pred_inv)
print(f'Test MSE for GROSS_SPA: {mse_gross}')
print(f'Test MAE for GROSS_SPA: {mae_gross}')

# Valutation NET_SPA
model_net.eval()
with torch.no_grad():
    y_net_pred = model_net(X_test.to(device)).cpu().numpy()
    y_net_test_inv = scaler_y_net.inverse_transform(y_net_test.cpu().numpy())
    y_net_pred_inv = scaler_y_net.inverse_transform(y_net_pred)

    # Inverse transformation
    y_net_test_inv = np.expm1(y_net_test_inv)  # inversa di log(1 + y)
    y_net_pred_inv = np.expm1(y_net_pred_inv)

# MSE and MAE for NET_SPA
mse_net = mean_squared_error(y_net_test_inv, y_net_pred_inv)
mae_net = mean_absolute_error(y_net_test_inv, y_net_pred_inv)
print(f'Test MSE for NET_SPA: {mse_net}')
print(f'Test MAE for NET_SPA: {mae_net}')
