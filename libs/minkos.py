import numpy as np
from minkfncts2d import MF2D

def calcul_fonctionelles(file1, max_treshold, resolution=100):
    F = []
    U = []
    Chi = []
    for threshold in np.linspace(0.0, max_treshold, resolution):
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


def condition_near_0(a,b,c):
    return a < 0.01 and b < 0.01 and c < 0.01



def get_start_end(f):
    s,e = 0,len(f)-1
    for i,v in enumerate(f):
        if i > 2:
            if condition_near_0(f[i-2],f[i-1],f[i]) and e == len(f)-1:
                e = i
    return s,e
    
def crop_functional(x,f):
    s,e = get_start_end(f)
    x = x[s:e]
    f = f[s:e]
    return x,f

def normalize_on_x(x,f): # start = 0.0, end = 40.0):
    s,e = x[0],x[-1]
    x,f = crop_functional(x,f)
    x = np.linspace(s, e, len(x))
    return x,f