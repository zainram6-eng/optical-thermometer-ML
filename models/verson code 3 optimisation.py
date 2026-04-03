import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Chargement des données
data = pd.read_excel("C:/Users/DELL/Desktop/5demis 2024-2025/TIPE 2025/experience/dataa.xlsx", sheet_name='Tabelle1').dropna()
data = data[(data['Température'].notna()) & (data['Biréfringence'].notna())]
X = data[['Température']]
y = data['Biréfringence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Pipeline et GridSearchCV - SVR
svr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])
svr_params = {
    'svr__kernel': ['rbf'],
    'svr__C': [1, 10, 100, 1000],
    'svr__gamma': ['scale', 0.1, 0.01],
    'svr__epsilon': [0.0001, 0.001, 0.01]
}
svr_grid = GridSearchCV(svr_pipeline, svr_params, cv=5, scoring='r2', n_jobs=-1)
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_

# 3. Pipeline et GridSearchCV - Random Forest
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # pas obligatoire pour RF, mais gardé pour uniformité
    ('rf', RandomForestRegressor(random_state=42))
])
rf_params = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, scoring='r2', n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# 4. Prédictions
y_pred_svr = best_svr.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

# 5. Évaluation
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
for model, metrics in results.items():
    print(f"\n{model.upper():^50}")
    print("-" * 50)
    print(f"R² score: {metrics['R²']:.4f}")
    print(f"MSE: {metrics['MSE']:.2e}")

# 6. Courbes de comparaison
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Données expérimentales')
plt.scatter(X_test, y_test, color='black', label='Test set')
plt.plot(X_test.sort_values('Température'),
         best_svr.predict(X_test.sort_values('Température')),
         color='blue', linewidth=2, label='SVR')
plt.plot(X_test.sort_values('Température'),
         best_rf.predict(X_test.sort_values('Température')),
         color='green', linewidth=2, label='Random Forest')
plt.xlabel("Température (°C)", fontsize=12)
plt.ylabel("Biréfringence (Δn)", fontsize=12)
plt.title("Modélisation de la biréfringence (SVR vs Random Forest)", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
