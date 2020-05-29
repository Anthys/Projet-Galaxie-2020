import numpy as np 
import matplotlib.pyplot as plt
from libs.pic_process import *
from scipy import signal

import argparse
import os,sys

def main(myFile):

    global args

    # Importer l'image dans file1. L'image doit être "normalisée" entre 0 et 255 comme d'hab avec coolrange

    file1, name, ext = get_image(myFile)
    file1 = np.clip(np.rint(file1), 0, 255)
    
    # Tracer l'histogramme de l'image : nb de pixels pour chaque valeur d'intensité (lissé)
    
    NbOccurs = []
    threshold = [i for i in range(256)]
    for i in threshold:
        Occurs_i = np.count_nonzero(file1 == i)
        NbOccurs.append(Occurs_i)
    
    kernel = signal.gaussian(28, 7)
    NbOccursSmooth = np.convolve(NbOccurs, kernel, mode='same')
    accroiss = np.diff(NbOccursSmooth, append=[0])
    accroissSmooth = np.convolve(accroiss, kernel, mode='same')

    size_window = [8,5]

    fig = plt.figure(figsize = (*size_window,))
    fig.add_subplot(121)
    plt.title("Galaxie")
    plt.imshow(file1, cmap="viridis")
    plt.colorbar()

    fig.add_subplot(122)
    plt.plot(threshold, NbOccursSmooth)
    plt.plot(threshold, accroissSmooth)
    plt.title("Histogramme des occurrences")
    plt.xlabel(r"Seuil $\nu$")
    plt.ylabel(r"Nombre d'occurences")
    plt.tight_layout()
   
    plt.show()

    # Trouver une approximation grossière du premier point d'inflexion (si besoin lisser la courbe)

    accroiss = np.diff(NbOccursSmooth)




parser = ""
args = ""

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file in format FITS or DAT', type=str)
    parser.add_argument("-dat", "--dat", action="store_true", help="Force the DAT format processing")
    args = parser.parse_args()

    args.fantom = True


if __name__ == "__main__":
    init_args()
    main(args.file)