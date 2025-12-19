import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Geometry setup
# -------------------------
def setup_rectangle(Lx=5.0, Ly=2.0, nx=50, ny=20):
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    nodes = np.arange(0, (nx + 1) * (ny + 1)).reshape((nx + 1, ny + 1))
    nnodes = (nx + 1) * (ny + 1)

    # Node coordinates
    X = np.zeros((nnodes, 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            X[nodes[i, j], 0] = x[i]
            X[nodes[i, j], 1] = y[j]

    # Element connectivity
    conn = []
    for j in range(ny):
        for i in range(nx):
            conn.append([nodes[i, j], nodes[i+1, j], nodes[i+1, j+1], nodes[i, j+1]])
    conn = np.array(conn, dtype=int)

    # Boundary conditions: fix left edge
    bcs = {nodes[0, j]: [0, 1] for j in range(ny + 1)}

    # Load: downward at right edge center
    P = 1.0
    forces = {nodes[-1, ny//2]: [0, -P]}

    return conn, X, bcs, forces

# -------------------------
# Material and element routines (same as before)
# -------------------------
def quad_stiffness(E, nu, coords):
    C = E / (1 - nu**2) * np.array([[1, nu, 0],
                                    [nu, 1, 0],
                                    [0, 0, (1 - nu)/2]])
    gp = 1/np.sqrt(3) * np.array([[-1,-1],[1,-1],[1,1],[-1,1]])
    Ke = np.zeros((8,8))
    for xi, eta in gp:
        Nxi = 0.25 * np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)]])
        Neta= 0.25 * np.array([[-(1-xi), -(1+xi), (1+xi), (1-xi)]])
        J = np.zeros((2,2))
        for i in range(4):
            J[0,0] += Nxi[0,i]*coords[i,0]
            J[0,1] += Nxi[0,i]*coords[i,1]
            J[1,0] += Neta[0,i]*coords[i,0]
            J[1,1] += Neta[0,i]*coords[i,1]
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Negative or zero Jacobian!")
        Jinv = np.linalg.inv(J)
        B = np.zeros((3,8))
        for i in range(4):
            dN = np.array([Nxi[0,i], Neta[0,i]])
            dNdX = Jinv @ dN
            B[:, 2*i:2*i+2] = [[dNdX[0], 0],
                               [0, dNdX[1]],
                               [dNdX[1], dNdX[0]]]
        Ke += B.T @ C @ B * detJ
    return Ke

def assemble_system(conn, X, bcs, forces, E=10.0, nu=0.3):
    n_nodes = X.shape[0]
    K = np.zeros((2*n_nodes, 2*n_nodes))
    F = np.zeros(2*n_nodes)
    for el in conn:
        coords = X[el]
        Ke = quad_stiffness(E, nu, coords)
        dof = []
        for n in el:
            dof.extend([2*n, 2*n+1])
        for i in range(8):
            for j in range(8):
                K[dof[i], dof[j]] += Ke[i,j]
    for n, f in forces.items():
        F[2*n] = f[0]
        F[2*n+1] = f[1]
    fixed = []
    for n, bc in bcs.items():
        if 0 in bc: fixed.append(2*n)
        if 1 in bc: fixed.append(2*n+1)
    free = np.setdiff1d(np.arange(2*n_nodes), fixed)
    Kff = K[np.ix_(free, free)]
    Ff = F[free]
    u = np.zeros(2*n_nodes)
    u[free] = np.linalg.solve(Kff, Ff)
    return u

def compute_von_mises(conn, X, u, E=10.0, nu=0.3):
    C = E / (1 - nu**2) * np.array([[1, nu, 0],
                                    [nu, 1, 0],
                                    [0, 0, (1-nu)/2]])
    vm = []
    for el in conn:
        coords = X[el]
        dof = []
        for n in el:
            dof.extend([2*n, 2*n+1])
        ue = u[dof]
        xi = eta = 0.0
        Nxi = 0.25 * np.array([[-(1-eta), (1-eta), (1+eta), -(1+eta)]])
        Neta= 0.25 * np.array([[-(1-xi), -(1+xi), (1+xi), (1-xi)]])
        J = np.zeros((2,2))
        for i in range(4):
            J[0,0] += Nxi[0,i]*coords[i,0]
            J[0,1] += Nxi[0,i]*coords[i,1]
            J[1,0] += Neta[0,i]*coords[i,0]
            J[1,1] += Neta[0,i]*coords[i,1]
        Jinv = np.linalg.inv(J)
        B = np.zeros((3,8))
        for i in range(4):
            dN = np.array([Nxi[0,i], Neta[0,i]])
            dNdX = Jinv @ dN
            B[:, 2*i:2*i+2] = [[dNdX[0], 0],
                               [0, dNdX[1]],
                               [dNdX[1], dNdX[0]]]
        sigma = C @ (B @ ue)
        s_vm = np.sqrt(sigma[0]**2 - sigma[0]*sigma[1] + sigma[1]**2 + 3*sigma[2]**2)
        vm.append(s_vm)
    return np.array(vm)

def plot_von_mises(conn, X, vm):
    cx = X[conn, 0].mean(axis=1)
    cy = X[conn, 1].mean(axis=1)
    plt.figure(figsize=(8,4))
    sc = plt.scatter(cx, cy, c=vm, cmap='viridis', s=20)
    plt.colorbar(sc, label='Von Mises stress')
    plt.title("Von Mises Stress - Rectangle")
    plt.axis('equal')
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    conn, X, bcs, forces = setup_rectangle()
    u = assemble_system(conn, X, bcs, forces)
    vm = compute_von_mises(conn, X, u)
    plot_von_mises(conn, X, vm)


