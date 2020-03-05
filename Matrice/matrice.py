__Author__ = "aillet"
__Filename__ = "matrice.py"
__Creationdate__ = "24/02/2020"

import os
import numpy as np
from pic_process import *
from minkos import *



matrice = []
list_lettre = []
list_line = []
all_files = os.listdir("/home/aillet/Bureau/Infromatique/Projet-Galaxie-2020/Matrice/txt")
all_files_bis = os.listdir("/home/aillet/Bureau/Infromatique/Projet-Galaxie-2020/Matrice/fits")
for i in range(len(all_files)):
    all_files[i] = "/home/aillet/Bureau/Infromatique/Projet-Galaxie-2020/Matrice/txt/" + all_files[i]
for i in range(len(all_files_bis)):
    all_files_bis[i] = "/home/aillet/Bureau/Infromatique/Projet-Galaxie-2020/Matrice/fits/" + all_files_bis[i]

for txt in all_files:
    f = open(txt)
    list_line = f.readlines()
    list_lettre += [list_line[21][3:len(list_line[21])-1]]
    list_lettre += [list_line[22][3:len(list_line[22])-1]]
    list_lettre += [list_line[23][3:len(list_line[23])-1]]
    f.close()


for lettre in range(len(list_lettre)):
    list_lettre[lettre] = float(list_lettre[lettre])

for i in range(len(all_files_bis)):
    a = get_image(all_files_bis[i],dat = True)
    ab = contrastLinear(a[0],70)
    F,U,Chi = calcul_fonctionelles(ab,256)
    matrice += [F + U + Chi + [list_lettre[i]] + [list_lettre[i+1]] + [list_lettre[i+2]]]

matrice = np.array(matrice)
np.save('matrice',matrice)







