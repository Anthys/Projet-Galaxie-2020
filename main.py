import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os,sys

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve


def contrast(file1):
    if False:
        file1 = np.float64(file1)
        file1 = ((file1 - 128) / 128) * (np.pi / 2)
        file1 = 1*np.tanh(file1) + 1
        file1 = (file1 * 128) / 1
    return file1
    
def contrastLinear(file1, value):
    file1 = value*file1
    return file1

def fantom(file1):
    file2 = []
    for i in range(len(file1)):
        file3 = []
        for c in range(len(file1[i])):
            if not np.isnan(file1[i][c]):
                file3 += [file1[i][c]]
        if file3 != []:
            file2 += [file3]
    return file2



def main(myFile):
    global args

    name = myFile.split("/")
    name = name[-1]
    if "." in name:
        name = name.split(".")
        name = name[-2]

    # Récupérer le fichier
    if args.dat:
        file1 = np.loadtxt(myFile)
        file1 = np.float64(file1)
    else:
        file1 = fits.getdata(myFile)
        file1 = np.float64(file1)
        
        # Enlever les pixels fantomes
        if args.fantom:
            file1 = fantom(file1)

    

    # Réhausser le contraste
    if args.contrast:
        file1 = contrast(file1)
    if args.contrastLinear:
        file1 = contrastLinear(file1, args.contrastLinear)
        
    # Determiner l'interval de definition de la fonction
    max_lin = 40
    if type(args.max) == int:
        max_lin = args.max

    # Lissage de la fonction
    if args.smooth:
        size_gauss = args.smooth
        img = file1
        img_zerod = img.copy()
        img_zerod[np.isnan(img)] = 0
        # It is a 9x9 array
        kernel = Gaussian2DKernel(size_gauss)#x_stddev=1)
        file1 = scipy_convolve(img, kernel, mode='same', method='direct')

    # Calcul des fonctionelles
    F = []
    U = []
    Chi = []
    for threshold in np.linspace(0.0, max_lin, 100):
        (f, u, chi) = MF2D(file1, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)

    # Visuels 

    fig = plt.figure(figsize = (8,5))
    fig.add_subplot(121)
    plt.title("Galaxy")
    plt.imshow(file1, cmap="viridis")

    fig.add_subplot(122)
    x = np.linspace(0.0, max_lin, 100)
    if args.normalize:
        plt.plot(x, np.array(F)/np.max(F), x, np.array(U)/np.max(U), x, np.array(Chi)/np.max(Chi))
    else:
        plt.plot(x, F, x, U, x, Chi)
    plt.title("2D Minkowski Functions")
    plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"], bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.xlabel("Threshold")
    plt.tight_layout()

    # Fin
    if args.save:
        print(name)
        plt.savefig(args.save + "/" +name +".png")
    else:
        plt.show()


parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in fits format', type=str)
    parser.add_argument("-o", dest="output", help="remove input spacing", type = str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cT", "--contrast",action="store_true", help="Contraste tangeante hyperbolique")
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = int)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-f", "--fantom", action="store_true", help="delete the NaN pixels")
    parser.add_argument("-dat", "--dat", action="store_true", help="file is in dat format")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth")
    parser.add_argument("-cool", "--cool", action="store_true", help="ContrastTangeante+Normalize+Fantom+Smooth")
    args = parser.parse_args()

    if args.cool:
        args.contrast = True
        args.normalize = True
        if not args.smooth:
            args.smooth = 1

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)
