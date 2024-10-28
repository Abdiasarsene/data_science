import numpy as np
from scipy.optimize import minimize
import pandas as pd

# Fonction de coût courant Ln(x, u)
def L(n, x, u):
    return u ** 2 - x - 1

# Fonction de récurrence fn(x, u)
def f(n, x, u):
    return x + u

# Fonction de coût terminal g(x)
def g(x):
    return 0

# Optimisation continue pour chaque étape
def optimize_continuous(L_func, f_func, V_next, n, x):
    def objective(u):
        return L_func(n, x, u) + V_next(f_func(n, x, u))[0]

    result = minimize(objective, x0=0)
    optimal_u = result.x[0]
    optimal_cost = result.fun
    return optimal_cost, optimal_u

# Fonction principale d'induction rétrograde
def backward_induction(x0, time_horizon):
    V = {}
    policy = {}

    def V_final(x):
        return optimize_continuous(L, f, lambda x: (g(x), None), time_horizon-1, x)

    V[time_horizon-1] = V_final(x0)
    
    for n in range(time_horizon-2, -1, -1):
        V[n] = optimize_continuous(L, f, lambda x: V[n+1], n, x0)

    for n in range(time_horizon):
        policy[n] = V[n][1]

    return V, policy

# Utilisation de la récurrence pour calculer les états à chaque étape
def compute_states(x0, policy, time_horizon):
    x_values = [x0]
    for t in range(time_horizon):
        next_x = f(t, x_values[-1], policy[t])
        x_values.append(next_x)
    return x_values

# Résolution du problème par induction rétrograde et stockage dans un DataFrame
def solve_problem(x0, time_horizon):
    V, policy = backward_induction(x0, time_horizon)
    x_values = compute_states(x0, policy, time_horizon)
    
    data = {
        'Étape': list(range(time_horizon + 1)),
        'État x': x_values,
        'Contrôle u': [policy[t] if t < time_horizon else None for t in range(time_horizon + 1)],
        'Valeur optimale V': [V[t][0] if t < time_horizon else None for t in range(time_horizon + 1)]
    }
    df = pd.DataFrame(data)
    
    return df

# Paramètres d'entrée
x0 = 0  # État initial
time_horizon = 4  # Définissez l'horizon temporel ici

# Résolution du problème et affichage des résultats dans un DataFrame
df_results = solve_problem(x0, time_horizon)
print(df_results)