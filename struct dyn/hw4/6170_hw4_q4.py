import numpy as np
from truss_setup import truss_setup # calls the truss class
from scipy.optimize import minimize

# set up
E = 70e9      # Young's modulus
rho = 2700    # Density
Amin = 1e-3   # Minimum area
tr, Le, A0 = truss_setup(E=E, rho=rho, Amin=Amin)

# using KS approx
def freq_objective(A):
    ks_min, _ = tr.ks_min_eigenvalue(A, ks_rho=100.0, k=5)
    return -ks_min  # negative because most optimizers minimize

res_freq = minimize(freq_objective, A0, bounds=[(Amin, 1.0) for _ in range(tr.nelems)],
                    method='SLSQP', options={'disp': True})

A_freq_opt = res_freq.x
ks_min_opt, _ = tr.ks_min_eigenvalue(A_freq_opt, ks_rho=100.0, k=5)
print("KS min frequency optimized areas:", A_freq_opt)
print("KS minimum eigenvalue (frequency^2):", ks_min_opt)

# --- Compliance minimization ---
def compliance_objective(A):
    K = tr.assemble_stiffness_matrix(A)
    # Apply BCs by reduction
    Kr = tr.reduce_matrix(K)
    # Compute displacement vector under given loads
    # For simplicity assume tr.f is the global load vector
    M = tr.assemble_mass_matrix(A)  # not strictly needed for compliance
    f_reduced = tr.f[tr.reduced]  
    # Solve Kr u = f
    u = np.linalg.solve(Kr.todense(), f_reduced)
    return float(f_reduced.T @ u)  # Compliance

res_compliance = minimize(compliance_objective, A0, bounds=[(Amin, 1.0) for _ in range(tr.nelems)],
                          method='SLSQP', options={'disp': True})

A_comp_opt = res_compliance.x
comp_val = compliance_objective(A_comp_opt)
print("Compliance optimized areas:", A_comp_opt)
print("Compliance value:", comp_val)

# --- Compare designs ---
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plt.plot(A_freq_opt, 'o-', label='Frequency-optimal')
plt.plot(A_comp_opt, 's-', label='Compliance-optimal')
plt.xlabel('Element index')
plt.ylabel('Cross-sectional area [m^2]')
plt.title('Truss Member Areas Comparison')
plt.legend()
plt.grid(True)
plt.show()
