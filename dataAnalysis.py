import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("resources/averageCleaned.csv")

# Crea un istogramma per GROSS_SPA
plt.figure(figsize=(10, 5))
sns.histplot(df['GROSS_SPA'], kde=True, bins=30)
plt.title('Distribuzione di GROSS_SPA')
plt.xlabel('GROSS_SPA')
plt.ylabel('Frequenza')
plt.savefig('diagrams/Distribuzione_GROSS_SPA.png')

# Crea un istogramma per NET_SPA
plt.figure(figsize=(10, 5))
sns.histplot(df['NET_SPA'], kde=True, bins=30)
plt.title('Distribuzione di NET_SPA')
plt.xlabel('NET_SPA')
plt.ylabel('Frequenza')
plt.savefig('diagrams/Distribuzione_NET_SPA.png')

# Crea un boxplot per GROSS_SPA
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['GROSS_SPA'])
plt.title('Boxplot di GROSS_SPA')
plt.savefig('diagrams/Boxplot_GROSS_SPA.png')

# Crea un boxplot per NET_SPA
plt.figure(figsize=(10, 5))
sns.boxplot(x=df['NET_SPA'])
plt.title('Boxplot di NET_SPA')
plt.savefig('diagrams/Boxplot_NET_SPA.png')
