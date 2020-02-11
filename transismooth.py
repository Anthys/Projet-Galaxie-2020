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

def smooth_file(file1, valsmooth):
    size_gauss = valsmooth
    img = file1
    img_zerod = img.copy()
    img_zerod[np.isnan(img)] = 0
    # It is a 9x9 array
    kernel = Gaussian2DKernel(size_gauss)#x_stddev=1)
    file1 = scipy_convolve(img, kernel, mode='same', method='direct')
    return file1


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

    fig = plt.figure(figsize = (8,5))
    fig.add_subplot(121)
    plt.title("Galaxy")
    plt.imshow(file1, cmap="viridis")


    fig.add_subplot(122)
    x = np.linspace(0.0, max_lin, 100)

    # Lissage de la fonction

    param = args.functional

    draw_all = args.drawall

    if args.smooth:
        F, U, Chi = calcul_fonctionelles(file1, max_lin)
        le_liste = [param]
        if draw_all:
            le_liste = ["f","u","chi"]
        for i in le_liste:
            h,col = get_right(F,U,Chi, i)
            b = np.max(h)
            if args.nonorm:
                b = 1
            plt.plot(x, np.array(h)/b, color=col + [1] )

        for i in range(1,args.smooth):
            a_values = np.linspace(0.01,1,args.smooth+1)
            temp_file1 = smooth_file(file1, i)
            F, U, Chi = calcul_fonctionelles(temp_file1, max_lin)
            for j in le_liste:
                h,col = get_right(F,U,Chi, j)
                b = np.max(h)
                if args.nonorm:
                    b = 1
                plt.plot(x, np.array(h)/b, color=col +[a_values[-i-1]] )
    
    plt.title("2D Minkowski Functionals")
    #plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"], bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.legend(le_liste, bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.xlabel("Threshold")
    plt.tight_layout()

    # Fin
    if args.save:
        if args.name:
            name = args.name
        print(name)
        plt.savefig(args.save + "/" +name +".png")
    else:
        plt.show()

def get_right(F,U,chi,arg):
    if arg == "f":
        return F, [1,0,0]
    elif arg == "u":
        return U, [0,1,0]
    elif arg == "chi":
        return chi, [0,0,1]

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


parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in fits format', type=str)
    parser.add_argument("-o", dest="output", help="remove input spacing", type = str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = int, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int)
    parser.add_argument("-dat", "--dat", action="store_true", help="file is in dat format")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth")
    parser.add_argument("-n", "--name", type = str, help="name of file")
    parser.add_argument("-f", "--functional", type = str, help="name of functional to show",choices=['f', 'u', 'chi'], default = "chi")
    parser.add_argument("-all", "--drawall", action="store_true",help="draw all functionals")
    parser.add_argument("-nonorm", "--nonorm", action="store_true",help="No normalisation")
    args = parser.parse_args()

    args.cool = True
    if args.cool:
        args.contrast = True
        args.normalize = True
        if not args.smooth:
            args.smooth = 1

    args.fantom = True

    #sys.exit(0)


if __name__ == "__main__":
    init_args()
    main(args.file)
