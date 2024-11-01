import pandas as pd

# 1. Carica il file CSV
file_path = "../../data/averageFinal.csv"
df = pd.read_csv(file_path)

num_rows = len(df)
print(f"Numero di righe nel dataset prima della pulizia: {num_rows}")

# 2. Elimina le colonne specifiche (sostituisci con le colonne che vuoi eliminare)
columns_to_drop = ["ID", "PLANNING_EXPIRES_AT", "PREVIOUS_FDR", "PROD_TOOL", "REPLACED_BY_FRD", "REPLACES_FDR", "REQUESTED_AT", "DOC_ID", "ID_1", "AP_NO", "AUTHOR_1", "CREF", "QUESTION_NO"
                   , "PROC_ID", "ID_2", "ID_3", "CONSIDERATION_IN_COMMITTEE", "CONSIDERATION_IN_COMMITTEE_M", "AM_ESTIMATION", "AM_TABLING_DEADLINE", "AM_TABLING_DEADLINE_M", "VOTE_IN_COMMITTEE", "VOTE_IN_COMMITTEE_M",
                   "VOTE_IN_PLENARY", "VOTE_IN_PLENARY_M", "PROC_ID_1", "COMMITTEE", "PRESENTATION_IN_COMMITTEE", "AM_TABLING_DEADLINE_E", "CONSIDERATION_IN_COMMITTEE_E", "VOTE_IN_COMMITTEE_E", "VOTE_IN_PLENARY_E",
                   "SESSION_NO", "LDI", "DOSSIER_NAME_1", "DOSSIER_NAME", "NET_SUBMITTED", "FDR_VERSION", "DOC_DATE", "PE_NO", "REQUESTER_CODE", "PROC_NAME"]  # Sostituisci con i nomi delle colonne da eliminare
df_cleaned = df.drop(columns=columns_to_drop, errors='ignore')  # `errors='ignore'` evita errori se la colonna non esiste

# 3. Controlla i valori nulli nelle colonne rimanenti
null_values = df_cleaned.isnull().sum()

# Stampa il numero di valori nulli per ogni colonna
print("Valori nulli nelle colonne rimanenti:")
print(null_values)

# 4. Risoluzione dei valori nulli (se necessario)
# Se vuoi rimuovere le righe con valori nulli, usa la funzione dropna()
df_cleaned = df_cleaned.dropna()  # Rimuove le righe con valori nulli
# Se invece vuoi riempire i valori nulli con una specifica strategia, puoi usare fillna()
# df_cleaned = df_cleaned.fillna(0)  # Esempio: sostituisce i valori nulli con 0

# 5. Salva il file pulito come CSV
cleaned_file_path = "../../data/cleanedExperimentForecast.csv"
df_cleaned.to_csv(cleaned_file_path, index=False)

print(f"Il file pulito è stato salvato come: {cleaned_file_path}")

# 6. Stampa il numero di righe del dataset dopo la pulizia
num_rows = len(df_cleaned)
print(f"Numero di righe nel dataset dopo la pulizia: {num_rows}")

