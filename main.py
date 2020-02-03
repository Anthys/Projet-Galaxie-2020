import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os,sys


def contrast(file1):
    file1 = np.float64(file1)
    file1 = ((file1 - 128) / 128) * (np.pi / 2)
    file1 = 1*np.tanh(file1) + 1
    file1 = (file1 * 128) / 1
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
    hdu = fits.PrimaryHDU()
    hdu.data = file2
    # print(file2)
    return file2



def main(myFile):
    global args

    # Modifier le lien de l'image pour l'ouvrir
    file1 = fits.getdata(myFile)
    if args.fantom:
        file1 = fantom(file1)
    if args.contrast:
        file1 = contrast(file1)
    file1 = np.float64(file1)
    name = myFile.split("/")
    name = name[-1]

    max_lin = 40
    if type(args.max) == int:
        max_lin = args.max

    F = []
    U = []
    Chi = []

    for threshold in np.linspace(0.0, max_lin, 100):
        (f, u, chi) = MF2D(file1, threshold)
        F.append(f)
        U.append(u)
        Chi.append(chi)

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
    if args.save:
        print(name[:-5])
        plt.savefig(name[:-5]+".png")
    else:
        plt.show()


parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in fits format', type=str)
    parser.add_argument("-o", dest="output", help="remove input spacing", type = str)
    parser.add_argument("-s", "--save",action="store_true", help="save result without showing")
    parser.add_argument("-c", "--contrast",action="store_true", help="with contrast")
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-f", "--fantom", action="store_true", help="delete the NaN pixels")
    args = parser.parse_args()


if __name__ == "__main__":
    init_args()
    main(args.file)
