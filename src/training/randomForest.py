import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Carica il dataset
file_path = "../../data/cleanedExperimentForecast.csv"
df = pd.read_csv(file_path)

# Stampa i valori massimi di GROSS_SPA e NET_SPA
print("Valore massimo di GROSS_SPA:", df['GROSS_SPA'].max())
print("Valore massimo di NET_SPA:", df['NET_SPA'].max())

# 2. Rimuovi le righe con valori di GROSS_SPA o NET_SPA maggiori di 500
df = df[(df['NET_SPA'] <= 500)]

# Identifica le feature categoriali (stringhe)
categorical_columns = df.select_dtypes(include=['object']).columns

# One-Hot Encoding delle feature categoriali
encoder = OneHotEncoder(drop='first', sparse_output=False)  # `drop='first'` per evitare multicollinearità
encoded_categorical = encoder.fit_transform(df[categorical_columns])

# Crea un DataFrame con le feature codificate
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_columns))

# Unisci le feature codificate al resto del dataset (rimuovendo le vecchie feature categoriali)
df_encoded = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)

# Separa feature (X) e target (y_gross e y_net)
X = df_encoded.drop(columns=['GROSS_SPA', 'NET_SPA'])  # Mantieni tutte le feature tranne il target
y_gross = df_encoded['GROSS_SPA']  # Target GROSS
y_net = df_encoded['NET_SPA']  # Target NET

# Dividi i dati in training set e test set (80% training, 20% test)
X_train, X_test, y_gross_train, y_gross_test = train_test_split(X, y_gross, test_size=0.2, random_state=42)
X_train_net, X_test_net, y_net_train, y_net_test = train_test_split(X, y_net, test_size=0.2, random_state=42)

# Modello Random Forest per GROSS_SPA
model_gross = RandomForestRegressor(random_state=42)
model_gross.fit(X_train, y_gross_train)

# Predizioni sui dati di test per GROSS_SPA
y_gross_pred = model_gross.predict(X_test)

# Valutazione delle performance del modello per GROSS_SPA
mse_gross = mean_squared_error(y_gross_test, y_gross_pred)
mae_gross = mean_absolute_error(y_gross_test, y_gross_pred)
r2_gross = r2_score(y_gross_test, y_gross_pred)

print(f"Test MSE for GROSS_SPA: {mse_gross}")
print(f"Test MAE for GROSS_SPA: {mae_gross}")
print(f"Test R² for GROSS_SPA: {r2_gross}")

# Importanza delle feature per GROSS_SPA
importances_gross = pd.Series(model_gross.feature_importances_, index=X.columns)
importances_gross = importances_gross.sort_values(ascending=False)

print("Importanza delle feature per GROSS_SPA:")
print(importances_gross)

# Modello Random Forest per NET_SPA
model_net = RandomForestRegressor(random_state=42)
model_net.fit(X_train_net, y_net_train)

# Predizioni sui dati di test per NET_SPA
y_net_pred = model_net.predict(X_test_net)

# Valutazione delle performance del modello per NET_SPA
mse_net = mean_squared_error(y_net_test, y_net_pred)
mae_net = mean_absolute_error(y_net_test, y_net_pred)
r2_net = r2_score(y_net_test, y_net_pred)

print(f"Test MSE for NET_SPA: {mse_net}")
print(f"Test MAE for NET_SPA: {mae_net}")
print(f"Test R² for NET_SPA: {r2_net}")

# Importanza delle feature per NET_SPA
importances_net = pd.Series(model_net.feature_importances_, index=X.columns)
importances_net = importances_net.sort_values(ascending=False)

print("Importanza delle feature per NET_SPA:")
print(importances_net)
