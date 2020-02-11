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

from pic_process import *
from minkos import *

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
        file1 = smooth_file(file1, args.smooth)

    # Calcul des fonctionelles
    F, U, Chi = calcul_fonctionelles(file1, max_lin)
    

    # Visuels
    
    size_window = [8,5]

    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(121)
    plt.title("Galaxy - "+name)
    plt.imshow(file1, cmap="viridis")

    fig.add_subplot(122)
    x = np.linspace(0.0, max_lin, 100)
    a,b,c = 1,1,1
    if args.normalize:
        a = np.max(F)
        b = np.max(U)
        c = np.max(Chi)
    plt.plot(x, np.array(F)/a, color = func_col("f"))
    plt.plot(x, np.array(U)/b, color = func_col("u"))
    plt.plot(x, np.array(Chi)/c, color = func_col("chi"))
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
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = int, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth", default=1)
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)
