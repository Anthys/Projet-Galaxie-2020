import numpy as np
from minkfncts2d import MF2D

def calcul_fonctionelles(file1, max_treshold):
    F = []
    U = []
    Chi = []
    for threshold in np.linspace(0.0, max_treshold, 100):
        (f, u, chi) = MF2D(file1, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)
    return F, U, Chi

def func_col(arg):
    if arg == "f":
        return [1,0,0]
    elif arg == "u":
        return [0,0.6,0]
    elif arg == "chi":
        return [0,0,1]
    else:
        return [0,0,0]