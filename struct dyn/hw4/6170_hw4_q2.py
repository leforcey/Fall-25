import numpy as np
from truss_setup import truss_setup   # function for the geometry since its the same for all the Qs
# --> truss_setup builds the truss (geometry, BCs, loads, material, initial areas)

# set everything up
tr, Le, A = truss_setup(E=70e9, rho=2700, Amin=1e-3)

# Compute KS approximation of minimum eigenvalue
ks_rho = 100.0
ks_min, ks_grad = tr.ks_min_eigenvalue(A, ks_rho=ks_rho, k=5)

min_freq = np.sqrt(ks_min)

print("KS-approx minimum eigenvalue (λ):", ks_min)
print("KS-approx minumum natural frequency (ω):", min_freq)
print("Gradient d(KS)/dA:")
print(ks_grad)

# finite difference check
eps = 1e-6
p = np.linspace(-1, 1, tr.nelems)

ks1, _ = tr.ks_min_eigenvalue(A, ks_rho=ks_rho)
ks2, _ = tr.ks_min_eigenvalue(A + eps*p, ks_rho=ks_rho)

fd = (ks2 - ks1) / eps
ans = np.dot(ks_grad, p)

print("\nFinite-difference verification:")
print("analytic:", ans)
print("FD      :", fd)
print("Rel err :", abs((ans - fd) / ans))

