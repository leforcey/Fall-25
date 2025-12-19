# ae 6170 homework 3
# question 2 and 3 code
# Lauren Forcey
# (i have two separate files but this also plots the mass vs iterations if you uncomment lines 138-145)

import numpy as np
import matplotlib.pyplot as plt

# QUESTION 2 CODE (literally copied, should've made this more modular oops):
# given
L = 2.5 # length of each bar
H = 2.5 # height of each bar
num_panels = 5  # 5 boxes in the truss
E = 70e9            # Pa
smax = 300e6        # Pa
rho = 2700          # kg/m^3
Amin = 1e-3         # m^2
P = 250e3           # N

# defining the truss geometry from the problem
# bottom row, y = 0 on the bottom
bottom_row = np.array([[i*L, 0] for i in range(num_panels + 1)])
# Top row, y = H on the top
top_row = np.array([[i*L, H] for i in range(num_panels + 1)])
# stack them for the full truss
nodes = np.vstack((bottom_row, top_row))
num_nodes = len(nodes)

# calculating the connectivity
# used chat to help verify implementation
elem = []
# bottom row
for i in range(num_panels):
    elem.append((i, i + 1))
# top row
for i in range(num_panels):
    elem.append((num_panels + 1 + i, num_panels + 2 + i))
# straight vertical lines
for i in range(num_panels + 1):
    elem.append((i, num_panels + 1 + i))
# X members going diagonally
for i in range(num_panels):
    n1 = i
    n2 = i + 1
    n3 = num_panels + 1 + i
    n4 = num_panels + 2 + i
    elem.append((n1, n4))
    elem.append((n2, n3))
num_bars = len(elem)

# calculate the lengths (using the distance formula)
lengths = np.array([np.linalg.norm(nodes[i]-nodes[j]) for (i,j) in elem])

# loads
forces = np.zeros(2*len(nodes))
for i in range(1, num_panels):
    forces[2*i + 1] = -P  # negative y-direction

# bcs (constrained in x and y on both corners)
fixed_dofs = [0, 1, 10, 11]

# initial area of each (creating list of them)
A = np.ones(num_bars) * Amin

# OK NOW 
# FSD STEP #1: compute the forces in each member, i for each load case, j (Fij(x))
def compute_forces(A):
    # used chat to determine what should be in this function
    K = np.zeros((2*num_nodes, 2*num_nodes))
    
    for idx, (i, j) in enumerate(elem):
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2)
        c = (xj - xi)/L
        s = (yj - yi)/L
        # element stiffness matrix
        k = (E*A[idx]/L) * np.array([
            [ c*c,  c*s, -c*c, -c*s], 
            [ c*s,  s*s, -c*s, -s*s],
            [-c*c, -c*s,  c*c,  c*s],
            [-c*s, -s*s,  c*s,  s*s]
        ])
        # creating global stiffness matrix
        dof = [2*i, 2*i+1, 2*j, 2*j+1]
        for m in range(4):
            for n in range(4):
                K[dof[m], dof[n]] += k[m, n]
    
    # Apply boundary conditions
    free_dofs = np.setdiff1d(np.arange(2*num_nodes), fixed_dofs)
    u = np.zeros(2*num_nodes)
    u[free_dofs] = np.linalg.solve(K[np.ix_(free_dofs, free_dofs)], forces[free_dofs])
    
    # force in each bar
    Fijx = np.zeros(num_bars)
    for idx, (i, j) in enumerate(elem):
        xi, yi = nodes[i]
        xj, yj = nodes[j]
        L = np.sqrt((xj - xi)**2 + (yj - yi)**2) # distance formulaa
        c = (xj - xi)/L
        s = (yj - yi)/L
        u_e = np.array([u[2*i], u[2*i+1], u[2*j], u[2*j+1]])
        Fijx[idx] = (E*A[idx]/L) * np.array([-c, -s, c, s]) @ u_e
    return Fijx

tol = 1e-6
max_iter = 100
# making list for all the values
mass_list = []

# now going through FSD procedure
# x = A (structural variable) from notes, given Amin
for iteration in range(max_iter):
    # STEP 1: compute the forces in each member
    Fijx = compute_forces(A)

    # STEP 2/3: compute minumum member sizes satisfying constraints
    # setting the lower/upper bound on the stress constraint (said to do)
    A_new = np.maximum(np.abs(Fijx)/smax, Amin)

    # STEP 4: calculate the mass with the new xi
    mass = np.sum(rho * A_new * lengths)
    mass_list.append(mass)
    
    # check convergence
    if np.all(np.abs(A_new - A)/A < tol):
        print(f"{iteration+1} iterations")
        break

    # update for the next iter
    A = A_new

min_mass = mass_list[-1]
print(f"Minumum mass: {min_mass:.2f} kg")

# plot that ish
#plt.figure()
#plt.plot(range(1,len(mass_list)+1), mass_list,'D-')
#plt.xlabel('Iteration number')
#plt.ylabel('Truss Mass (kg)')
#plt.title('Mass vs Iterations')

#plt.grid(True)
#plt.show()


# QUESTION 3 CODE
# implementing the KS function
def c_ks(s, rho):
    c = np.max(s)
    return c + np.log(np.sum(np.exp(rho*(s-c))))/rho

def dc_ks(s, rho):
    c = np.max(s)
    eta = np.exp(rho*(s-c))
    eta /= np.sum(eta)
    return eta

# now implement FD and CS from the notes
def finite_difference(f, s, rho, h=1e-6):
    n = len(s)
    grad_fd = np.zeros(n)
    for i in range(n):
        sph = s.copy()
        sph[i] += h
        grad_fd[i] = (f(sph, rho) - f(s, rho)) / h
    return grad_fd

def complex_step(f, s, rho, h=1e-30):
    n = len(s)
    grad_cs = np.zeros(n)
    for i in range(n):
        spih = s.astype(complex)
        spih[i] += 1j * h
        grad_cs[i] = np.imag(f(spih, rho)) / h
    return grad_cs

# Compute stresses and normalized stress ratios
# scaling stuff that dr kennedy talked about in class
Fijx = compute_forces(A)
stresses = np.abs(Fijx) / A
s_ratios = stresses / smax
rho_ks = 50  # started with 10, increased until smoother plot for CS
cKS = c_ks(s_ratios, rho_ks)
grad_adjoint = dc_ks(s_ratios, rho_ks)

# implement all the methods to have stuff to plot
h_values = np.logspace(-12, 0, 40) # e-12 to e0
error_fd = []
error_cs = []
for h in h_values:
    grad_fd = finite_difference(c_ks, s_ratios, rho_ks, h=h)
    grad_cs = complex_step(c_ks, s_ratios, rho_ks, h=h)
    err_fd = np.linalg.norm(grad_fd - grad_adjoint)
    err_cs = np.linalg.norm(grad_cs - grad_adjoint)
    error_fd.append(err_fd)
    error_cs.append(err_cs)


# Plotting all of it
plt.figure()
plt.loglog(h_values, error_fd, 'bo-', label='FD - adj')
plt.loglog(h_values, error_cs, 'gD-', label='CS - adj')
plt.xlabel('Step size, h')
plt.ylabel('Normalized error')
plt.title('Adjoint vs Forward Difference (FD) and Complex step (CS)')
plt.grid(True, which='both')
plt.legend()
plt.show()
print(f"\nAverage FD error: {error_fd[ np.argmin(np.abs(h_values-1e-6)) ]:.2e}")
print(f"Average CS error: {error_cs[0]:.2e}")
