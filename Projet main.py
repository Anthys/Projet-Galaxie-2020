__Author__ = "aillet"
__Filename__ = "Projet main.py"
__Creationdate__ = "16/01/2020"

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D



# Modifier le lien de l'image pour l'ouvrir
file = fits.getdata("/home/aillet/Bureau/Infromatique/ProjetS4/minkfncts2d-master/4.fits")

def contrast_fantome(file1,name):
    def getvariablename(vara):
        for k in globals():
            if globals()[k] == vara:
                return k
    file1 = np.float64(file1)
    file1 = ((file1 - 128) / 128) * (np.pi / 2)
    file1 = 3 * np.tanh(file1) + 3
    file1 = (file1 * 128) / 3
    file2 = []
    for i in range(len(file1)):
        file3 = []
        for c in range(len(file1[i])):
            if not np.isnan(file1[i][c]):
                file3 += [file1[i][c]]
        if file3 != []:
            file2 += [file3]
    nam = 'cf_' + name + '.fits'
    hdu = fits.PrimaryHDU()
    hdu.data = file2
    hdu.writeto(nam)  #creer un fichier fits tout bien comme on veut




def main(file_crop,name):
    F = []
    U = []
    Chi = []

    contrast_fantome(file_crop,name)
    name_file = 'cf_' + name + '.fits'
    data = fits.getdata(name_file)
    data = np.float64(data)

    for threshold in np.linspace(0.0, 255, 300):
        (f, u, chi) = MF2D(data, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)

    plt.figure(1)
    plt.clf()
    plt.imshow(data, cmap="viridis")
    plt.show()

    plt.figure(1)
    plt.clf()
    x = np.linspace(0.0, 255, 300)
    plt.plot(x, F, x, U, x, Chi)
    plt.title("2D Minkowski Functions")
    plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"])
    plt.xlabel("Threshold")
    plt.show()


main(file,'5')