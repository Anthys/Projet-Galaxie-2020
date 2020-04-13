
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from minkfncts2d import MF2D
import argparse
import os,sys

#sys.path.insert(1, 'libs/')

from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

from libs.pic_process import *
from libs.minkos import *

def main(myFile):
    global args

    #file1, name, ext = get_image(myFile)
    name = myFile.split("/")[1].split(".")[0]
    file1,name = charger_le_putain_de_fichier(myFile)
    #print(file1)
    #sys.exit()

    # Réhausser le contraste
    if args.contrastLinear:
        file1 = contrastLinear(file1, args.contrastLinear)
        
    # Determiner l'interval de definition de la fonction
    max_lin = 40
    if type(args.max) == int:
        max_lin = args.max

    # Lissage de la fonction
    
    

    # Visuels
    
    size_window = [10,8]

    fig = plt.figure(figsize = (*size_window,))
    
    
    fig.add_subplot(121)
    plt.title("Galaxy - "+name)
    print(file1)
    plt.imshow(file1, cmap="viridis")

    fig.add_subplot(122)
    x = np.linspace(0.0, max_lin, 100)
    a,b,c = 1,1,1
    if args.normalize:
        a = coef_normalization_functional(F)
        b = coef_normalization_functional(U)
        c = coef_normalization_functional(Chi)
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

    for i,par in enumerate([0]):
        pars = [221,222, 223, 224]
        part = pars[i]
        plt.title("Galaxy")
        plt.imshow(file1, cmap="viridis")
        plt.colorbar()
        if False:
            txtsss = ["0","pi_by_6","pi_by_4","pi_by_3"]
            if 1:
                file2 = rotation_X(file1,par)
                txt = "smooth_"+str(i-1)
            fig.add_subplot(part)
            #plt.title("Galaxy - "+name)
            plt.title("Rotation by theta = " + txtsss[i])
            print(file1)
            plt.imshow(file2, cmap="viridis")
        elif False:
            txt = ""
            part = pars[i]
            txtsss = ["pi_by_6","pi_by_4","pi_by_3"]
            if True:
                file2 = rotation_X(file1,par)
                txt = "rotation_"+str(txtsss[i])
            else:
                file2 = file1
                txt = "sans_alteration"

            # Calcul des fonctionelles
            resolution = 256
            F, U, Chi = calcul_fonctionelles(file2, max_lin, resolution)
            fig.add_subplot(part)
            x = np.linspace(0.0, max_lin, resolution)
            a,b,c = 1,1,1
            if args.normalize:
                a = np.max(np.abs(F))
                b = np.max(np.abs(U))
                c = np.max(np.abs(Chi))
            
            file_w = open('sofianejetaime/' + name + "_"+txt+".out", "w+")
            file_w.write("# Threshold then F then U then Chi"+ "\n")
            file_w.write(str(list(x)) + "\n")
            file_w.write(str(F) + "\n")
            file_w.write(str(U) + "\n")
            file_w.write(str(Chi) + "\n")
            file_w.close()
            if True:
                plt.plot(x, np.array(F)/a, color = func_col("f"))
                plt.plot(x, np.array(U)/b, color = func_col("u"))
                plt.plot(x, np.array(Chi)/c, color = func_col("chi"))
                plt.title("2D Minkowski Functions" + " smooth lvl:" + str(i-1))
    plt.legend(["F (Area)", "U (Boundary)", "$\chi$ (Euler characteristic)"], bbox_to_anchor =(1,-0.2), loc = "upper right")
    plt.xlabel("Threshold")
    plt.tight_layout()
    plt.savefig("sofianejetaime/colorbar.png")
    #plt.show()

    sys.exit()
    for i,par in enumerate([221,222, 223, 224]):
        if i == 0:
            fig.add_subplot(221)
            plt.title("Galaxy - "+name)
            print(file1)
            plt.imshow(file1, cmap="viridis")
        else:
            txt = ""
            if i -1> 0:
                file2 = smooth_file(file1, i-1)
                txt = "smooth_"+str(i-1)
            else:
                file2 = file1
                txt = "sans_alteration"

            # Calcul des fonctionelles
            resolution = 256
            F, U, Chi = calcul_fonctionelles(file2, max_lin, resolution)
            fig.add_subplot(par)
            x = np.linspace(0.0, max_lin, resolution)
            a,b,c = 1,1,1
            if args.normalize:
                a = np.max(np.abs(F))
                b = np.max(np.abs(U))
                c = np.max(np.abs(Chi))
            
            file_w = open('sofianejetaime/' + name + "_"+txt+".out", "w+")
            file_w.write("# Threshold then F then U then Chi"+ "\n")
            file_w.write(str(list(x)) + "\n")
            file_w.write(str(F) + "\n")
            file_w.write(str(U) + "\n")
            file_w.write(str(Chi) + "\n")
            file_w.close()

            plt.plot(x, np.array(F)/a, color = func_col("f"))
            plt.plot(x, np.array(U)/b, color = func_col("u"))
            plt.plot(x, np.array(Chi)/c, color = func_col("chi"))
            plt.title("2D Minkowski Functions" + " smooth lvl:" + str(i-1))



parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = float, default=0.00000045)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int, default=255)
    parser.add_argument("-n", "--normalize", action="store_true", help="normalize the curves")
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth", default=1)
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)
