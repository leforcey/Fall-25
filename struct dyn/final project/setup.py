import numpy as np
from mainscript import PlaneStress
from filters import NodeFilter
from mass_compliance import MassConsstrainedCompliance

def rectangular_setup(nx=125, ny=50, Lx=5.0, Ly=2.0, P=1.0, area_fraction=0.4):
    """
    Set up a rectangular domain for topology optimization.
    Returns: ps, fltr, opt, x_init, lb, ub, nnodes
    """
    return _generic_setup(nx, ny, Lx, Ly, P, area_fraction, shape='rect')

def lbracket_setup(nx=125, ny=50, Lx=5.0, Ly=2.0, P=1.0, area_fraction=0.4):
    """
    Set up an L-bracket domain for topology optimization.
    Returns: ps, fltr, opt, x_init, lb, ub, nnodes
    """
    return _generic_setup(nx, ny, Lx, Ly, P, area_fraction, shape='lbracket')

def _generic_setup(nx, ny, Lx, Ly, P, area_fraction, shape='rect'):
    # Node coordinates
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    nodes = np.arange((nx + 1) * (ny + 1)).reshape((nx + 1, ny + 1))
    
    nnodes = (nx + 1) * (ny + 1)
    nelems = nx * ny
    
    X = np.zeros((nnodes, 2))
    for j in range(ny + 1):
        for i in range(nx + 1):
            X[nodes[i, j], 0] = x[i]
            X[nodes[i, j], 1] = y[j]
    
    # Connectivity
    conn = np.zeros((nelems, 4), dtype=int)
    for j in range(ny):
        for i in range(nx):
            conn[i + j * nx, 0] = nodes[i, j]
            conn[i + j * nx, 1] = nodes[i + 1, j]
            conn[i + j * nx, 2] = nodes[i + 1, j + 1]
            conn[i + j * nx, 3] = nodes[i, j + 1]
    
    # Boundary conditions
    bcs = {}
    for j in range(ny):
        bcs[nodes[0, j]] = [0, 1]
    
    # Load
    forces = {}
    forces[nodes[-1, 0]] = [0, -P]
    
    # Optionally mask elements for L-bracket
    if shape == 'lbracket':
        for j in range(ny//2):           # remove upper right quarter
            for i in range(nx//2, nx):
                idx = i + j * nx
                conn[idx, :] = -1       # mark as inactive
    
    # Spatial filter
    r0 = 2 * (Lx / nx)
    fltr = NodeFilter(conn, X, r0=r0, ftype="spatial")
    
    # Plane stress problem
    ps = PlaneStress(conn, X, bcs, forces, fltr=fltr)
    
    # Optimization object
    opt = MassConsstrainedCompliance(ps, area_fraction * Lx * Ly)
    
    # Initial design
    x_init = area_fraction * np.ones(nnodes)
    lb = 1e-3 * np.ones(nnodes)
    ub = np.ones(nnodes)
    
    return ps, fltr, opt, x_init, lb, ub, nnodes
