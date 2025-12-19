import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# =============================================================
# PARAMETERS
# =============================================================
nelx, nely = 100, 60
volfrac = 0.40
penal   = 3.0
E0      = 1.0
Emin    = 1e-9
nu      = 0.3

move_limit = 0.15
p_norm = 12.0        # Le et al. use a high p to approximate max-stress

# =============================================================
# MESH AND DOMAIN (L-shape)
# =============================================================
nnx, nny = nelx + 1, nely + 1
nn = nnx * nny
ndof = nn * 2

def node(i, j):
    return j * nnx + i

# L-beam: cut lower-right quadrant
def in_L_shape(i, j):
    return not (i > nelx//2 and j < nely//2)

mask = np.array([[in_L_shape(i,j) for j in range(nely)] for i in range(nelx)], bool)

# Element connectivity
conn = []
for i in range(nelx):
    for j in range(nely):
        if not mask[i, j]:
            continue
        n1=node(i,j); n2=node(i+1,j); n3=node(i+1,j+1); n4=node(i,j+1)
        conn.append([n1,n2,n3,n4])
conn = np.array(conn)
ne = len(conn)

# =============================================================
# QUAD4 ELEMENT STIFFNESS (PLANE STRESS)
# =============================================================
def quad4_Bmat(xi, eta):
    """Shape function derivatives at Gauss point."""
    dN_dxi = 0.25*np.array([
        [-(1-eta),  (1-eta), (1+eta), -(1+eta)],
        [-(1-xi ), -(1+xi ), (1+xi ),  (1-xi )]
    ])
    return dN_dxi

def strain_displacement_matrix(dNdx):
    B = np.zeros((3,8))
    B[0,0::2] = dNdx[0];
    B[1,1::2] = dNdx[1];
    B[2,0::2] = dNdx[1];
    B[2,1::2] = dNdx[0];
    return B

# material matrix
D = (E0 / (1-nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1-nu)/2]
])

# precompute element stiffness with fictitious density 1.0
Ke_list = []
Be_list = []

# build element matrices
for e, nodes in enumerate(conn):
    # element coordinates
    coords = np.array([[i % nnx, i//nnx] for i in nodes], float)
    Ke = np.zeros((8,8))
    Bstore = []
    gauss = [(-1/np.sqrt(3), -1/np.sqrt(3)),
             ( 1/np.sqrt(3), -1/np.sqrt(3)),
             ( 1/np.sqrt(3),  1/np.sqrt(3)),
             (-1/np.sqrt(3),  1/np.sqrt(3))]

    for (xi, eta) in gauss:
        dN_dxi = quad4_Bmat(xi, eta)
        J = dN_dxi @ coords
        detJ = np.linalg.det(J)
        dNdx = np.linalg.solve(J.T, dN_dxi).T

        B = strain_displacement_matrix(dNdx)
        Ke += B.T @ (D @ B) * detJ
        Bstore.append((B, detJ))

    Ke_list.append(Ke)
    Be_list.append(Bstore)

# =============================================================
# BOUNDARY CONDITIONS
# clamp left edge
# =============================================================
fix = []
for j in range(nny):
    i = 0
    n = node(i,j)
    fix.append(2*n)
    fix.append(2*n+1)
fix = np.array(fix)

# load at top right
load = np.zeros(ndof)
topnode = node(nelx, nely)
load[2*topnode+1] = -1.0   # downward

# =============================================================
# FE SOLVE
# =============================================================
def FE(x):
    K = lil_matrix((ndof, ndof))
    for e, nodes in enumerate(conn):
        rho = x[e]**penal
        Ke = Emin*Ke_list[e] + rho*(Ke_list[e])
        dofs = np.r_[np.array(nodes)*2, np.array(nodes)*2+1]
        for i in range(8):
            for j in range(8):
                K[dofs[i], dofs[j]] += Ke[i,j]
    K = csr_matrix(K)

    u = np.zeros(ndof)
    free = np.setdiff1d(np.arange(ndof), fix)
    u[free] = spsolve(K[free][:,free], load[free])
    return u

# =============================================================
# STRESS & p-NORM AGGREGATION
# =============================================================
def element_stresses(u, x):
    vm = np.zeros(ne)
    for e, nodes in enumerate(conn):
        dofs = np.r_[np.array(nodes)*2, np.array(nodes)*2+1]
        ue = u[dofs]

        # integrate stresses at Gauss points, average
        s_avg = 0
        for (B, detJ) in Be_list[e]:
            strain = B @ ue
            stress = (Emin + x[e]**penal*(E0-Emin)) * strain
            sx, sy, txy = stress
            vm_g = np.sqrt(sx**2 + sy**2 - sx*sy + 3*txy**2)
            s_avg += vm_g
        vm[e] = s_avg / 4.0
    return vm

def pnorm_stress(vm):
    return (np.sum(vm**p_norm))**(1/p_norm)

# =============================================================
# INITIALIZE DENSITY
# =============================================================
x = volfrac * np.ones(ne)

# =============================================================
# OPTIMIZATION LOOP (MMA-style density update)
# =============================================================
for it in range(50):
    u = FE(x)
    vm = element_stresses(u, x)
    f = pnorm_stress(vm)

    # sensitivity df/dx
    df = (vm**(p_norm-1)) / (f**(p_norm-1))
    df *= penal * x**(penal-1)  # SIMP derivative

    # MMA-style bounded update
    x_new = x - 0.05 * df / (np.max(np.abs(df)) + 1e-9)
    x_new = np.maximum(0.001, np.minimum(1.0, x_new))

    # volume constraint
    scale = volfrac * ne / np.sum(x_new)
    x_new *= scale

    # move limit
    x = np.maximum(x - move_limit, np.minimum(x + move_limit, x_new))

    print(f"Iter {it:02d}: p-norm stress = {f:.4f}")

# =============================================================
# PLOTTING (tricontour, magma)
# =============================================================
# generate node coordinates
X, Y = np.meshgrid(np.arange(nnx), n)
