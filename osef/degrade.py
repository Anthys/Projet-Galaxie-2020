import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os, sys
from math import floor

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

# from libs.pic_process import *
from libs.minkos import *


# file2 = fits.open("my_file.fits")

def abime(file2):
    # file2 = get_image(my_file)
    img = file2
    img_zerod = img.copy()
    # img_zerod[np.isnan(img)] = 0
    kernel = np.array([[1 / 81 for i in range(9)] for j in range(9)])
    # kernel = Gaussian2DKernel(1)  # x_stddev=1)
    file2 = scipy_convolve(img, kernel, mode='same', method='direct')
    return file2


def contrast(file1):
    """
      A ne pas utiliser pour le moment, on doit trouver une meilleure fonction mathématique
    """
    return file1
    """file1 = np.float64(file1)
    file1 = ((file1 - 128) / 128) * (np.pi / 2)
    file1 = 1*np.tanh(file1) + 1
    file1 = (file1 * 128) / 1
    return file1"""


def contrastLinear(file1, value):
    file1 = value * file1
    return file1


def fantom(file1):
    """
      Enlève les pixels fantomes créés par le crop de ds9
    """
    file2 = []
    for i in range(len(file1)):
        file3 = []
        for c in range(len(file1[i])):
            if not np.isnan(file1[i][c]):
                file3 += [file1[i][c]]
        if file3 != []:
            file2 += [file3]
    file2 = np.float64(file2)
    return file2


def smooth_file(file1, valsmooth):
    """
      Lisse une image avec une matrice gaussiene
    """
    size_gauss = valsmooth
    img = file1
    img_zerod = img.copy()
    # img_zerod[np.isnan(img)] = 0
    # It is a 9x9 array
    kernel = Gaussian2DKernel(size_gauss)  # x_stddev=1)
    file1 = scipy_convolve(img, kernel, mode='same', method='direct')
    return file1


def get_dat_file(name):
    file1 = np.loadtxt(name)
    file1 = np.float64(file1)
    return file1


def get_fit_file(name):
    file1 = fits.getdata(name)
    file1 = np.float64(file1)
    return file1


def get_image(path, dat=False):
    """
      Obtenir une image en format np-array-64, son nom et son extension
    """
    name = path.split("/")
    name = name[-1]
    ext = ""

    if "." in name:
        name = name.split(".")
        ext = name[-1]
        name = name[-2]

    # Récupérer le fichier
    if dat or ext == "dat":
        file1 = get_dat_file(path)
    elif ext == "fits":
        file1 = get_fit_file(path)
        # Enlever les pixels fantomes
        file1 = fantom(file1)
    return file1, name, ext


def abime2(file1):
    img1 = np.float64(file1)
    img2 = []
    for i in range(floor(img1.shape[0]/2)):
        img2 += [[]]
        for j in range(floor(img1.shape[1]/2)):
            moyenne = (img1[i*2][j*2] + img1[i*2 + 1][j*2] + img1[i*2][j*2 + 1] + img1[i*2 + 1][j*2 + 1])/4
            img2[i] += [moyenne]
    img2 = np.float64(img2)
    return img2



def main(myFile):
    global args

    file1, name, ext = get_image(myFile)

    # Réhausser le contraste
    if args.contrastLinear:
        file1 = contrastLinear(file1, args.contrastLinear)

    # Determiner l'interval de definition de la fonction
    max_lin = 40
    if type(args.max) == int:
        max_lin = args.max

    # Lissage de la fonction
    if args.smooth:
        # file1 = smooth_file(file1, args.smooth)
        # file1 = abime(file1)
        file1 = abime2(file1)


    # Calcul des fonctionelles
    F, U, Chi = calcul_fonctionelles(file1, max_lin)

    # Visuels

    size_window = [8, 5]

    fig = plt.figure(figsize=(*size_window,))
    fig.add_subplot(121)
    plt.title("Galaxy - " + name)
    plt.imshow(file1, cmap="viridis")

    fig.add_subplot(122)
    x = np.linspace(0.0, max_lin, 100)
    a, b, c = 1, 1, 1
    if args.normalize:
        a = np.max(F)
        b = np.max(U)
        c = np.max(Chi)
    plt.plot(x, np.array(F) / a, color=func_col("f"))
    plt.plot(x, np.array(U) / b, color=func_col("u"))
    plt.plot(x, np.array(Chi) / c, color=func_col("chi"))
    plt.title("2D Minkowski Functions")
    plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"], bbox_to_anchor=(1, -0.2),
               loc="upper right")
    plt.xlabel("Threshold")
    plt.tight_layout()

    # Fin
    if args.save:
        print(name)
        plt.savefig(args.save + "/" + name + ".png")
    else:
        plt.show()


parser = ""
args = ""


def init_args():
    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type=float, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type=int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    parser.add_argument("-smooth", "--smooth", type=int, help="smooth", default=1)
    args = parser.parse_args()

    args.fantom = True


init_args()
main(args.file)

# print(abime("O98_HRS204_oooNET.fits"))
# print(smooth_file(get_image("O98_HRS204_oooNET.fits"), 1))
# print(np.shape(get_image("O98_HRS204_oooNET.fits")[0]))

# def degradeFile(my_file):
#     my_file = np.float64(my_file)
#     return my_file

# file.close()

# f = open("/home/raphael/PycharmProjects/Projet-Galaxie-2020/Matrice/txt/NGC1300_HAWK-I-r+g+b.dat.txt", 'r')
# traiter_dat("/home/raphael/PycharmProjects/Projet-Galaxie-2020/Matrice/txt/NGC1300_HAWK-I-r+g+b.dat.txt")
# print(f.read())
# f.close()
