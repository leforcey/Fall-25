import numpy as np
from truss import Truss   # inport the truss class from the notes

""" outputs:
        tr: Truss
        E_len: element lengths (array)
        Ai_vec: Amin (initial area vector)
"""

def truss_setup(E=70e9, rho=2700, Amin=1e-3):
    
    L = 2.5 # length of each bar
    num_panels = 5 # number of panels (5 boxes)

    # defining the nodes
    pos = []
    for i in range(num_panels+1):
        pos.append([i*L, 0])
    for i in range(num_panels+1):
        pos.append([i*L, L])
    pos = np.array(pos)

    # connectivity
    conn = []

    # bottom row
    for i in range(num_panels):
        conn.append([i, i+1])

    # top row
    offset = num_panels + 1
    for i in range(num_panels):
        conn.append([offset+i, offset+i+1])

    # verticals
    for i in range(num_panels+1):
        conn.append([i, offset+i])

    # diagonals
    for i in range(num_panels):
        conn.append([i, offset+i+1])
        conn.append([offset+i, i+1])

    conn = np.array(conn, dtype=int)

    # bcs
    left = 0
    right = num_panels
    bcs = np.array([
        2*left, 2*left+1,
        2*right, 2*right+1
    ], dtype=int)

    # forces
    f = np.zeros(2*pos.shape[0])

    # length of the elements
    xd = pos[conn[:,1],0] - pos[conn[:,0],0]
    yd = pos[conn[:,1],1] - pos[conn[:,0],1]
    E_len = np.sqrt(xd**2 + yd**2)

    # now create the truss
    tr = Truss(conn, pos, bcs, f, E=E, rho=rho)

    # Initial areas
    Ai_vec = Amin * np.ones(conn.shape[0])

    return tr, E_len, Ai_vec

