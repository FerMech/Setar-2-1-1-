import numpy as np
import matplotlib.pyplot as plt

# Paramètres théoriques
phi_1_0, phi_2_0 = 0.2, 0.2  # Intercepts des deux régimes
phi_1_1, phi_2_1 = 0.5, -0.5  # Coefficients autorégressifs
s = 0.0  # Seuil
sigma = 1  # Écart-type du bruit
T = 50  # Nombre de périodes
N_sim = 1000  # Nombre de simulations

# Stocker les valeurs estimées
phi_1_0_emp = []
phi_1_1_emp = []
phi_2_0_emp = []
phi_2_1_emp = []

np.random.seed(42)

# Simulation de N_sim séries
for n in range(N_sim):
    y = np.zeros(T)
    y[0] = np.random.normal(0, sigma)  # Condition initiale
    
    for t in range(1, T):
        epsilon = np.random.normal(0, sigma)
        if y[t-1] <= s:
            y[t] = phi_1_0 + phi_1_1 * y[t-1] + epsilon
        else:
            y[t] = phi_2_0 + phi_2_1 * y[t-1] + epsilon


    # Extraction des régimes
    regime_1_idx = np.where(y[:-1] <= s)[0]  # Indices du régime 1
    regime_2_idx = np.where(y[:-1] > s)[0]  # Indices du régime 2

    # Estimation des paramètres manuellement
    if len(regime_1_idx) > 1:
        X1 = np.column_stack((np.ones(len(regime_1_idx)), y[regime_1_idx]))        
        y1 = y[regime_1_idx + 1]
        beta1 = np.linalg.inv(X1.T @ X1) @ (X1.T @ y1)
        phi_1_0_emp.append(beta1[0])
        phi_1_1_emp.append(beta1[1])
    
    if len(regime_2_idx) > 1:
        X2 = np.column_stack((np.ones(len(regime_2_idx)), y[regime_2_idx]))  # Ajouter une constante
        y2 = y[regime_2_idx + 1]
        beta2 = np.linalg.inv(X2.T @ X2) @ X2.T @ y2
        phi_2_0_emp.append(beta2[0])
        phi_2_1_emp.append(beta2[1])
plt.plot(y)        

# Moyennes et écarts-types des paramètres estimés
mean_phi_1_0 =  np.mean(phi_1_0_emp)
print(phi_1_0_emp)
std_phi_1_0  =  np.std(phi_1_0_emp)
mean_phi_1_1 = np.mean(phi_1_1_emp)
std_phi_1_1  =  np.std(phi_1_1_emp)
mean_phi_2_0 = np.mean(phi_2_0_emp)
std_phi_2_0  =  np.std(phi_2_0_emp)
mean_phi_2_1 = np.mean(phi_2_1_emp)
std_phi_2_1  =  np.std(phi_2_1_emp)

# Affichage des résultats
print(f"Paramètres théoriques vs. Moyennes des simulations ({N_sim} simulations) :")
print(f"------------------------------------------------")
print(f"Paramètre    | Théorique | Moyenne Simulée | Écart-type")
print(f"------------------------------------------------")
print(f"phi_1_0      | {phi_1_0:.2f}      | {mean_phi_1_0:.2f}          | {std_phi_1_0:.2f}")
print(f"phi_1_1      | {phi_1_1:.2f}      | {mean_phi_1_1:.2f}          | {std_phi_1_1:.2f}")
print(f"phi_2_0      | {phi_2_0:.2f}      | {mean_phi_2_0:.2f}          | {std_phi_2_0:.2f}")
print(f"phi_2_1      | {phi_2_1:.2f}      | {mean_phi_2_1:.2f}          | {std_phi_2_1:.2f}")

# Visualisation des distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
params = [(phi_1_0_emp, phi_1_0, "phi_1_0"), (phi_1_1_emp, phi_1_1, "phi_1_1"),
          (phi_2_0_emp, phi_2_0, "phi_2_0"), (phi_2_1_emp, phi_2_1, "phi_2_1")]

for ax, (param_values, true_value, label) in zip(axes.flatten(), params):
    ax.hist(param_values, bins=30, alpha=0.7, color="blue", edgecolor="black", density=True)
    ax.axvline(x=true_value, color='red', linestyle='dashed', linewidth=2, label=f"Théorique: {true_value}")
    ax.set_title(f"Distribution de {label}")
    ax.set_xlabel(f"Valeurs estimées de {label}")
    ax.set_ylabel("Densité")
    ax.legend()

plt.tight_layout()
plt.show()
