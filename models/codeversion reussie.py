import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ================================
# 1. Chargement des données Excel
# ================================
# Charger le fichier Excel en sautant les lignes vides
data = pd.read_excel("C:/Users/DELL/Desktop/5demis 2024-2025/TIPE 2025/experience/datah.xlsx", sheet_name='Tabelle1').dropna()

# Nettoyage des données (suppression des lignes avec valeurs manquantes)
data = data[(data['Température'].notna()) & (data['Biréfringence'].notna())]

# ================================
# 2. Préparation des données
# ================================
X = data[['Température']]
y = data['Biréfringence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 3. Pipeline SVR optimisé
# ================================
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.001))
])
svr_pipe.fit(X_train, y_train)
y_pred_svr = svr_pipe.predict(X_test)

# ================================
# 4. Pipeline Random Forest optimisé
# ================================
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42))
])
rf_pipe.fit(X_train, y_train)
y_pred_rf = rf_pipe.predict(X_test)

# ================================
# 5. Évaluation robuste
# ================================
results = {
    'SVR': {
        'R²': r2_score(y_test, y_pred_svr),
        'MSE': mean_squared_error(y_test, y_pred_svr)
    },
    'Random Forest': {
        'R²': r2_score(y_test, y_pred_rf),
        'MSE': mean_squared_error(y_test, y_pred_rf)
    }
}

# ================================
# 6. Visualisation améliorée
# ================================
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Toutes les données')
plt.scatter(X_test, y_test, color='black', label='Test set')
plt.plot(X_test.sort_values('Température'),
         svr_pipe.predict(X_test.sort_values('Température')),
         color='blue', linewidth=2, label='SVR')
plt.plot(X_test.sort_values('Température'),
         rf_pipe.predict(X_test.sort_values('Température')),
         color='green', linewidth=2, label='Random Forest')
plt.xlabel("Température (°C)", fontsize=12)
plt.ylabel("Biréfringence (Δn)", fontsize=12)
plt.title("Prédiction de la biréfringence en fonction de la température", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# ================================
# 7. Affichage clair des résultats
# ================================
print("\n" + "="*50)
print("RÉSULTATS D'ÉVALUATION".center(50))
print("="*50)

for model, metrics in results.items():
    print(f"\n{model.upper():^50}")
    print("-"*50)
    print(f"R² score: {metrics['R²']:.4f}")
    print(f"MSE: {metrics['MSE']:.2e}")







































