# Convertido automaticamente de notebook Jupyter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('marketing.csv')
print(df.head())
print()
df.info()

print(df.describe())

plt.figure(figsize=(8,5))
plt.scatter(df['anuncios'], df['cliques'], color='steelblue', alpha=0.6)
plt.xlabel('Número de Anúncios')
plt.ylabel('Número de Cliques')
plt.title('Dispersão: Anúncios x Cliques')
plt.tight_layout()
plt.savefig('dispersao_marketing.png', dpi=120)
plt.show()
print("Correlação:", df['anuncios'].corr(df['cliques']).round(4))

X = df[['anuncios']]
y = df['cliques']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

modelo = LinearRegression()
modelo.fit(X_train, y_train)

print(f"Intercepto (b0): {modelo.intercept_:.4f}")
print(f"Coeficiente (b1): {modelo.coef_[0]:.4f}")
print(f"Equação: cliques = {modelo.intercept_:.2f} + {modelo.coef_[0]:.2f} * anuncios")

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, color='steelblue', alpha=0.6, label='Dados reais')
plt.plot(X_test, modelo.predict(X_test), color='red', linewidth=2, label='Reta de regressão')
plt.xlabel('Número de Anúncios')
plt.ylabel('Número de Cliques')
plt.title('Regressão Linear – Anúncios vs Cliques')
plt.legend()
plt.tight_layout()
plt.savefig('regressao_linear.png', dpi=120)
plt.show()

from sklearn.metrics import r2_score, mean_squared_error
y_pred = modelo.predict(X_test)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

