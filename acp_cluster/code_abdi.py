#Code

import numpy as np
from scipy.optimize import minimize

# Fonction de coût courant Ln(x, u)
def L(n, x, u):
    """
    Calcule le coût courant à l'étape n.
   
    Arguments :
    - n : étape temporelle
    - x : état courant
    - u : contrôle (réel)
   
    Retourne : coût courant
    """
    return (x - u) ** 2 + n * u ** 2

# Fonction de récurrence fn(x, u)
def f(n, x, u):
    """
    Calcule l'état suivant selon la récurrence f.
   
    Arguments :
    - n : étape temporelle
    - x : état courant
    - u : contrôle (réel)
   
    Retourne : état suivant
    """
    return x + u

# Fonction de coût terminal g(x)
def g(x):
    """
    Calcule le coût terminal.
   
    Argument :
    - x : état final
   
    Retourne : coût terminal
    """
    return 0

# Optimisation continue pour chaque étape
def optimize_continuous(L_func, f_func, V_next, n, x):
    """
    Minimise la fonction de coût en continu sur l'ensemble des actions réelles (u).
   
    Arguments :
    - L_func : fonction de coût courant
    - f_func : fonction de récurrence
    - V_next : fonction valeur de l'étape suivante
    - n : étape temporelle
    - x : état courant
   
    Retourne : coût minimum et contrôle optimal u
    """
    def objective(u):
        """Fonction à minimiser : L(n, x, u) + V(n+1, f(n, x, u))"""
        return L_func(n, x, u) + V_next(f_func(n, x, u))[0]

    # Minimisation sur l'ensemble des réels (u)
    result = minimize(objective, x0=0)  # Estimation initiale u = 0
    optimal_u = result.x[0]  # Contrôle optimal
    optimal_cost = result.fun  # Coût minimal
    return optimal_cost, optimal_u

# Fonction principale d'induction rétrograde
def backward_induction(x0, n):
    """
    Résout le problème d'optimisation dynamique par induction rétrograde.
   
    Arguments :
    - x0 : état initial
    - n : horizon temporel (nombre d'étapes)
   
    Retourne : dictionnaire des fonctions valeur et politique optimale.
    """
    V = {}  # Dictionnaire pour stocker la fonction valeur pour chaque étape
    policy = {}  # Dictionnaire pour stocker la politique optimale (u optimal)

    # Fonction pour calculer la fonction valeur à l'étape finale V(n-1, x)
    def V_n_minus_1(x):
        return optimize_continuous(L, f, lambda x: (g(x), None), n-1, x)

    # Génération des fonctions V(t, x) pour les étapes intermédiaires de t = n-2 à 0
    def V_t(x, t):
        return optimize_continuous(L, f, lambda x: V[t+1](x), t, x)

    # Stockage de la fonction valeur pour l'étape finale
    V[n-1] = V_n_minus_1

    # Calcul et stockage de V(t, x) pour chaque étape de n-2 à 0
    for t in reversed(range(n-1)):
        V[t] = (lambda x, t=t: V_t(x, t))  # Capture correcte de la variable t

    # Calcul des politiques optimales (contrôle optimal) à chaque étape
    for t in range(n):
        policy[t] = V[t](x0)[1]

    return V, policy

# Utilisation de la récurrence pour calculer les états à chaque étape
def compute_states(x0, policy, n):
    """
    Calcule les états à chaque étape en utilisant la relation de récurrence.
   
    Arguments :
    - x0 : état initial
    - policy : dictionnaire des contrôles optimaux (politique)
    - n : horizon temporel (nombre d'étapes)
   
    Retourne : liste des états successifs.
    """
    x_values = [x0]  # Liste pour stocker les états successifs
    for t in range(n):  # Calcul des états successifs
        next_x = f(t, x_values[-1], policy[t])
        x_values.append(next_x)
    return x_values

# Demander à l'utilisateur d'entrer l'horizon temporel
x0 = 0  # État initial
n = int(input("Entrez l'horizon temporel (nombre d'étapes) : "))  # Demander à l'utilisateur d'entrer n

# Résolution du problème par induction rétrograde
V, policy = backward_induction(x0, n)

# Affichage des fonctions valeur et des politiques optimales
print("\nFonctions valeur pour chaque étape :")
for t in range(n):
    V_value, u_optimal = V[t](x0)
    print(f"Étape {t} : V({t}, x0) = {V_value}, u optimal = {u_optimal}")

# Calcul et affichage des états successifs
x_values = compute_states(x0, policy, n)
print("\nÉtats calculés :")
for t, x in enumerate(x_values):
    print(f"x{t} = {x}")