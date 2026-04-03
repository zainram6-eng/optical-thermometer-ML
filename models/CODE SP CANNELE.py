



import numpy as np
from matplotlib import pyplot as plt
# Données
x = np.array([6, 5, 4, 3, 2, 1])
y = np.array([0.002244669,0.0020503,0.001919886,0.001749169,0.00157903,0.001410437])
# Régression linéaire
coefficients = np.polyfit(x, y, 1)
polynomial = np.poly1d(coefficients)
y_fit = polynomial(x)
# Affichage des points
plt.plot(x, y, 'o', color='red', label='Points expérimentaux')
# Affichage de la droite de régression
plt.plot(x, y_fit, '-', color='blue', label=f'Régression linéaire: y = {coefficients[0]:.2e}x + {coefficients[1]:.2e}')
# Configuration du graphique
plt.xlabel('ordre p')
plt.ylabel('longueur d\'onde -1')
plt.grid(True)
plt.legend()
plt.title('Régression linéaire')
# Affichage
plt.show()
# Affichage des coefficients dans la console
print(f"Pente (a): {coefficients[0]}")
print(f"Ordonnée à l'origine (b): {coefficients[1]}")
print(f"Équation de la droite: y = {coefficients[0]:.2e}x + {coefficients[1]:.2e}")