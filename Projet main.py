__Author__ = "aillet"
__Filename__ = "Projet main.py"
__Creationdate__ = "16/01/2020"

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D


# Modifier le lien de l'image pour l'ouvrir
file1 = fits.getdata("/home/aillet/Bureau/Infromatique/programmation/ProjetS4/photo/1.fits")
file1 = np.float64(file1)

F = []
U = []
Chi = []

for threshold in np.linspace(0.0, 1.0, 100):
    (f, u, chi) = MF2D(file1, threshold)
    F.append(f)
    U.append(u)
    Chi.append(chi)

plt.figure(1)
plt.clf()
plt.title("Image with Gaussian Noise")
plt.imshow(file1, cmap="viridis")
plt.show()

plt.figure(1)
plt.clf()
x = np.linspace(0.0, 1.0, 100)
plt.plot(x, F, x, U, x, Chi)
plt.title("2D Minkowski Functions")
plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"])
plt.xlabel("Threshold")
plt.show()
