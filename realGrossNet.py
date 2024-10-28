import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer

# Caricamento del file Excel
file_path = "resources/realGrossNet.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

# Sostituisci i valori non numerici con NaN
df['Average Gross'] = pd.to_numeric(df['Average Gross'], errors='coerce')
df['Average Net'] = pd.to_numeric(df['Average Net'], errors='coerce')

# Sostituzione dei valori mancanti con una categoria dummy
df['Procedure'].fillna('Unknown', inplace=True)
df['Committee'].fillna('Unknown', inplace=True)  # Esempio se altre colonne hanno valori nulli
df['Document Type'].fillna('Unknown', inplace=True)

# Rimuovi le righe con valori mancanti
df = df.dropna()

# 1. Esplorazione dei dati
print("Head:")
print(df.head())

# 2. Visualizzazione delle distribuzioni
numeric_columns = ['Gross Pages', 'Net Pages']


def cluster_gross_with_features(df, numeric_feature, categorical_features):
    # Preprocessing: Standardizzazione delle feature numeriche e One-Hot Encoding delle feature categoriali
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [numeric_feature]),
            ('cat', OneHotEncoder(drop='first'), categorical_features)
        ])

    # Applica il preprocessing
    df_preprocessed = preprocessor.fit_transform(df)

    # K-Means clustering
    kmeans = KMeans(n_clusters=5, random_state=42)  # Puoi scegliere il numero di cluster ottimale
    df['Cluster'] = kmeans.fit_predict(df_preprocessed)

    # Visualizzazione dei cluster
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=df[numeric_feature], y=df['Cluster'], hue=df['Cluster'], palette='viridis')
    title = f"Cluster su {numeric_feature} considerando {' e '.join(categorical_features)}"
    plt.title(title)
    plt.xlabel(numeric_feature)
    plt.ylabel("Cluster")
    plt.tight_layout()
    plt.savefig(f'diagrams/{numeric_feature}_{"_".join(categorical_features)}_cluster.png')

    # Stampa dei centri dei cluster
    print(f"Centri dei cluster ({', '.join(categorical_features)}):")
    print(kmeans.cluster_centers_)


# Eseguire il clustering con diverse combinazioni di feature categoriali
cluster_gross_with_features(df, 'Gross Pages', ['Committee'])
cluster_gross_with_features(df, 'Gross Pages', ['Procedure'])
cluster_gross_with_features(df, 'Gross Pages', ['Document Type'])
cluster_gross_with_features(df, 'Gross Pages', ['Committee', 'Procedure'])
cluster_gross_with_features(df, 'Gross Pages', ['Committee', 'Document Type'])
cluster_gross_with_features(df, 'Gross Pages', ['Procedure', 'Document Type'])
cluster_gross_with_features(df, 'Gross Pages', ['Committee', 'Procedure', 'Document Type'])

# Grafici di distribuzione per le variabili categoriali
plt.figure(figsize=(12, 6))
sns.countplot(x='Procedure', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Procedure')
plt.tight_layout()
plt.savefig('diagrams/procedure')

plt.figure(figsize=(20, 6))
sns.countplot(x='Committee', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Committee')
plt.tight_layout()
plt.savefig('diagrams/committee')

plt.figure(figsize=(12, 6))
sns.countplot(x='Document Type', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Document Type')
plt.tight_layout()
plt.savefig('diagrams/document_type')

for col in numeric_columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribuzione di {col}")
    plt.savefig('diagrams/gross_net_distribution')

# Clustering
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_columns])

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

print("Centri dei cluster:")
print(kmeans.cluster_centers_)

# iduzione della dimensionalità per la visualizzazione dei cluster
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster'], palette='viridis')
plt.title("Cluster visualizzati con PCA")
plt.savefig('diagrams/gross_net_cluster')

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm')
plt.title("Matrice di correlazione")
plt.savefig('diagrams/correlationMatrix')

########
#Boxplot
########
# Crea un boxplot per GROSS_SPA
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Gross Pages'])
plt.title('Boxplot di GROSS_SPA')
plt.savefig('diagrams/Boxplot_gross.png')

# Crea un boxplot per NET_SPA
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['Net Pages'])
plt.title('Boxplot di NET_SPA')
plt.savefig('diagrams/Boxplot_net.png')

plt.figure(figsize=(10, 5))
sns.boxplot(x='Procedure', y='Gross Pages', data=df)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('diagrams/boxplot_procedure_gross')

plt.figure(figsize=(20, 5))
sns.boxplot(x='Committee', y='Net Pages', data=df)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('diagrams/boxplot_committee_net')
