import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"CUDA is available. Using {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')
    print("CUDA is not available. Using CPU.")

# Load dataset
df = pd.read_csv("data/averageCleaned.csv")

categorical_columns = ["DOC_TYPE", "PROC_TYPE", "DOSSIER_TYPE"]

# Winsorization (optional)
# df['GROSS_SPA'] = np.clip(df['GROSS_SPA'], None, 550)
# df['NET_SPA'] = np.clip(df['NET_SPA'], None, 350)

# Filter out outliers
filtered_df = df[(df['GROSS_SPA'] <= 550) & (df['NET_SPA'] <= 350)]

# Dividing feature and target
X = filtered_df.drop(["GROSS_SPA", "NET_SPA", "ID"], axis=1)
y_gross = filtered_df["GROSS_SPA"]
y_net = filtered_df["NET_SPA"]

# One-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_columns)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)
X = X.toarray()

# Normalize features
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

# Logarithmic transformation for skewed data
y_gross = np.log1p(y_gross.values)
y_net = np.log1p(y_net.values)

# Normalize target variables
scaler_y_gross = MinMaxScaler()
y_gross = scaler_y_gross.fit_transform(y_gross.reshape(-1, 1))

scaler_y_net = MinMaxScaler()
y_net = scaler_y_net.fit_transform(y_net.reshape(-1, 1))

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y_gross = torch.tensor(y_gross, dtype=torch.float32)
y_net = torch.tensor(y_net, dtype=torch.float32)

# Define Neural Network Model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.3)  # Dropout
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out

# Hyperparameters
hidden_size1 = 32
hidden_size2 = 16
num_epochs = 100
learning_rate = 0.001
batch_size = 128

# Cross-Validation Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store results for each fold
mse_gross_scores = []
mae_gross_scores = []
mse_net_scores = []
mae_net_scores = []

# Cross-validation loop
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    print(f"Fold {fold+1}/{kf.get_n_splits()}")

    X_train, X_test = X[train_index], X[test_index]
    y_gross_train, y_gross_test = y_gross[train_index], y_gross[test_index]
    y_net_train, y_net_test = y_net[train_index], y_net[test_index]

    # Create datasets and data loaders
    train_dataset_gross = TensorDataset(X_train, y_gross_train)
    train_loader_gross = DataLoader(dataset=train_dataset_gross, batch_size=batch_size, shuffle=True)

    train_dataset_net = TensorDataset(X_train, y_net_train)
    train_loader_net = DataLoader(dataset=train_dataset_net, batch_size=batch_size, shuffle=True)

    # Define models
    model_gross = NeuralNetwork(X.shape[1], hidden_size1, hidden_size2, 1).to(device)
    model_net = NeuralNetwork(X.shape[1], hidden_size1, hidden_size2, 1).to(device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()  # MAE
    optimizer_gross = optim.Adam(model_gross.parameters(), lr=learning_rate)
    optimizer_net = optim.Adam(model_net.parameters(), lr=learning_rate)

    # Training loop for GROSS_SPA
    model_gross.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader_gross:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer_gross.zero_grad()
            outputs = model_gross(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer_gross.step()

            epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Loss for GROSS_SPA: {epoch_loss / len(train_loader_gross):.6f}')

    # Training loop for NET_SPA
    model_net.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader_net:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer_net.zero_grad()
            outputs = model_net(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            optimizer_net.step()

            epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss for NET_SPA: {epoch_loss / len(train_loader_net):.6f}')

    # Evaluation for GROSS_SPA
    model_gross.eval()
    with torch.no_grad():
        y_gross_pred = model_gross(X_test.to(device)).cpu().numpy()
        y_gross_test_inv = scaler_y_gross.inverse_transform(y_gross_test.cpu().numpy())
        y_gross_pred_inv = scaler_y_gross.inverse_transform(y_gross_pred)
        y_gross_test_inv = np.expm1(y_gross_test_inv)
        y_gross_pred_inv = np.expm1(y_gross_pred_inv)

        # Compute metrics
        mse_gross = mean_squared_error(y_gross_test_inv, y_gross_pred_inv)
        mae_gross = mean_absolute_error(y_gross_test_inv, y_gross_pred_inv)
        mse_gross_scores.append(mse_gross)
        mae_gross_scores.append(mae_gross)

    # Evaluation for NET_SPA
    model_net.eval()
    with torch.no_grad():
        y_net_pred = model_net(X_test.to(device)).cpu().numpy()
        y_net_test_inv = scaler_y_net.inverse_transform(y_net_test.cpu().numpy())
        y_net_pred_inv = scaler_y_net.inverse_transform(y_net_pred)
        y_net_test_inv = np.expm1(y_net_test_inv)
        y_net_pred_inv = np.expm1(y_net_pred_inv)

        # Compute metrics
        mse_net = mean_squared_error(y_net_test_inv, y_net_pred_inv)
        mae_net = mean_absolute_error(y_net_test_inv, y_net_pred_inv)
        mse_net_scores.append(mse_net)
        mae_net_scores.append(mae_net)

# Print average results
print(f'Cross-validated MSE for GROSS_SPA: {np.mean(mse_gross_scores)}')
print(f'Cross-validated MAE for GROSS_SPA: {np.mean(mae_gross_scores)}')
print(f'Cross-validated MSE for NET_SPA: {np.mean(mse_net_scores)}')
print(f'Cross-validated MAE for NET_SPA: {np.mean(mae_net_scores)}')
