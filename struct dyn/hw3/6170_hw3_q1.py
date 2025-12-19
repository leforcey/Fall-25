# homework 3
import numpy as np
import matplotlib.pyplot as plt

# ks function from the notes
def ks_func(s, rho):
    return np.max(s) + np.log(np.sum(np.exp(rho*(s - np.max(s)))))/rho

# list of si values
si_list = np.random.uniform(0.6, 1.0, 5)

# range of rho values (arbitrary)
rhos = np.linspace(1,100,50)

err_list = []

for rho in rhos:
    cks_eval = ks_func(si_list , rho)
    error = cks_eval - np.max(si_list)
    err_list.append(error)

err_list = np.array(err_list)

# plot the error vs 1/rho
plt.figure(figsize=(8,5))
plt.plot(1/rhos, err_list, marker='D')
plt.xlabel(r'$1/\rho$')
plt.ylabel(r'$c_{KS}(s, \rho) - \max_i(s_i)$')
plt.title(r'KS function error vs $1/\rho$' + '\n' + r'Lauren Forcey', fontsize=14)
plt.grid(True)
plt.show()