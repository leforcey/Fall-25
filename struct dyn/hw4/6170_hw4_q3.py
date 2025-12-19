import numpy as np
from scipy.optimize import minimize
from truss_setup import truss_setup

# setup
E = 70e9
rho_mat = 2700
Amin = 1e-3
Amax = 0.1 # just picked smth reasonable
tr, Le, A = truss_setup(E=E, rho=rho_mat, Amin=Amin)

# pick for compatible inequality constraints
M_max = 10000 # picking arbitrary mass greater than initial

# objective (KS implementation)
def objective(A):
    ks_min, _ = tr.ks_min_eigenvalue(A, ks_rho=100.0, k=5)
    omega_min = np.sqrt(ks_min)
    return -omega_min # minimize the negative to maximize the positive

# mass constraint
def mass_constraint(A):
    mass = np.sum(rho_mat * Le * A)
    return M_max - mass  # must be >= 0

constraints = {'type': 'ineq', 'fun': mass_constraint}
bounds = [(Amin, Amax) for x in range(tr.nelems)]

# minimize (ik we need to maximize, accounted for in the objective function)
res = minimize(objective, A, method='SLSQP', bounds=bounds, constraints=constraints,
               options={'disp': True, 'ftol': 1e-6, 'maxiter': 200})

# grab the results
A_opt = res.x
ks_min_opt, _ = tr.ks_min_eigenvalue(A_opt, ks_rho=100.0, k=5)
omega_min_opt = np.sqrt(ks_min_opt)

print("Maximum minimum natural frequency (rad/s):", omega_min_opt)


