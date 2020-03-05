__Author__ = "aillet"
__Filename__ = "Manipulation.py"
__Creationdate__ = "05/03/2020"

import numpy as np
matrice = np.load('/home/aillet/Bureau/Infromatique/Projet-Galaxie-2020/Matrice/matrice.npy')
matrice = np.rot90(matrice)

D_mat = []
D = np.sum(matrice,axis=1)
for i in range(len(D)):
    D[i] = D[i]/len(matrice[i])

for i in range (len(matrice)):
    D_mat += [D]
D_mat = np.array(D_mat)
D_mat = np.rot90(D_mat)


matrice = matrice - D_mat
