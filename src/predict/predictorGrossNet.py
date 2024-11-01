import joblib
import torch
import numpy as np
import pandas as pd

from src.models.neural_network_model import ComplexNeuralNetwork

# Carica gli scaler e il preprocessor
preprocessor = joblib.load('/home/daniele/PycharmProjects/AIStuff/models/preprocessor.pkl')
scaler_X = joblib.load('/home/daniele/PycharmProjects/AIStuff/models/scaler_X.pkl')
scaler_y_gross = joblib.load('/home/daniele/PycharmProjects/AIStuff/models/scaler_y_gross.pkl')
scaler_y_net = joblib.load('/home/daniele/PycharmProjects/AIStuff/models/scaler_y_net.pkl')

# Carica il modello GROSS_SPA
model_gross = torch.load('/home/daniele/PycharmProjects/AIStuff/models/full_model_gross.pth', map_location=torch.device('cpu'))
model_gross.eval()

# Carica il modello NET_SPA
model_net = torch.load('/home/daniele/PycharmProjects/AIStuff/models/full_model_net.pth', map_location=torch.device('cpu'))
model_net.eval()

def predict_gross_net(new_data):
    # Creiamo un DataFrame per i nuovi dati
    new_df = pd.DataFrame([new_data])

    # Applica One-Hot Encoding con il preprocessor caricato
    new_data_encoded = preprocessor.transform(new_df).toarray()

    # Normalizza i nuovi dati con lo scaler caricato
    new_data_normalized = scaler_X.transform(new_data_encoded)

    # Converte i dati in tensor di PyTorch
    new_data_tensor = torch.tensor(new_data_normalized, dtype=torch.float32)

    # Predizione per GROSS_SPA
    with torch.no_grad():
        y_gross_pred = model_gross(new_data_tensor).cpu().numpy()
        y_gross_pred_inv = scaler_y_gross.inverse_transform(y_gross_pred)
        y_gross_pred_final = np.expm1(y_gross_pred_inv).flatten()[0]  # Trasformazione inversa del log

    print(f'Predizione GROSS_SPA: {y_gross_pred_final}')

    # Predizione per NET_SPA
    with torch.no_grad():
        y_net_pred = model_net(new_data_tensor).cpu().numpy()
        y_net_pred_inv = scaler_y_net.inverse_transform(y_net_pred)
        y_net_pred_final = np.expm1(y_net_pred_inv).flatten()[0]  # Trasformazione inversa del log

    print(f'Predizione NET_SPA: {y_net_pred_final}')

# Esempio di nuovi dati per la predizione
new_data = {
    "DOC_TYPE": "RR",
    "PROC_NATURE": "INIT",
    "PROC_TYPE": "INI",
    "ROLE": "MAIN",
    "DOSSIER_TYPE": "FEMM",
    "COMMITTEE_1": "FEMM"
}

# Fare la predizione
predict_gross_net(new_data)
