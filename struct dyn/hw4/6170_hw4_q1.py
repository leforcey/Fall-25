import numpy as np
from truss import Truss   # same truss class ive been using
from truss_setup import truss_setup

# set everything up
tr, Le, A = truss_setup(E=70e9, rho=2700, Amin=1e-3)

# calculate the natural frequencies
omega, phi = tr.frequencies(A, k=5)
print("Natural frequencies (rad/s):")
for i, w in enumerate(omega):
    print(f"  mode {i+1}: {w:.6f}")

# calculate the derivatives
analytic = tr.frequency_derivative(A, k=5)
print("\nDerivative of the natural frequencies wrt area:")
print(analytic[0])

# finite difference check
eps = 1e-6
p = np.linspace(-1, 1, tr.nelems)

omega1, _ = tr.frequencies(A, k=5)
omega2, _ = tr.frequencies(A + eps*p, k=5)

fd = (omega2 - omega1)/eps
ans = np.dot(analytic[0], p)

print("\nFinite-difference test:")
print("Analytic:      ", ans)
print("Finite diff:   ", fd[0])
print("Rel error:     ", abs((ans - fd[0]) / ans))


