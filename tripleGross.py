import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Caricamento del dataset
file_path = "resources/realGrossNet.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

df.dropna(subset=['Procedure', 'Committee', 'Document Type'], inplace=True)

# Convertiamo 'Gross Pages' in numerico per assicurarci che non ci siano problemi con i dati
df['Gross Pages'] = pd.to_numeric(df['Gross Pages'], errors='coerce')

# Rimuovi i NaN che potrebbero essere rimasti
df.dropna(subset=['Gross Pages'], inplace=True)

# Creiamo una nuova colonna combinata con le triple di valori
df['Triple'] = df['Committee'] + ' | ' + df['Procedure'] + ' | ' + df['Document Type']

# Contiamo la frequenza di ogni combinazione
top_20_triples = df['Triple'].value_counts().nlargest(40).index

triple_counts = df['Triple'].value_counts().nlargest(40)

# Stampiamo le occorrenze delle 20 combinazioni più frequenti
print("Conteggio delle 20 combinazioni più frequenti:")
print(triple_counts)

# Filtra i dati solo per le 20 combinazioni più frequenti
df_filtered = df[df['Triple'].isin(top_20_triples)]

# Visualizza un boxplot per mostrare la distribuzione dei valori di Gross Pages per le 20 triple più frequenti
plt.figure(figsize=(20, 10))
sns.boxplot(x='Triple', y='Gross Pages', data=df_filtered, order=top_20_triples)
plt.xticks(rotation=90, ha='right')
plt.title("Distribuzione di Gross Pages per le 40 Triple più frequenti (Committee, Procedure, Document Type)")
plt.tight_layout()
plt.savefig('diagrams/triple')
