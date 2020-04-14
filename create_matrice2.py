import os,sys,argparse
import numpy as np
from libs.pic_process import *
from libs.minkos import *
from libs.matrices3 import *

import matplotlib.pyplot as plt
import matplotlib
from copy import copy

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("images_path", help='', type=str)
    parser.add_argument("-p", "--process", help='',action="store_true")
    parser.add_argument("-n", "--no_treat", help="Pas de traitement et renomage des fichiers (si vous l'avez deja fait)",action="store_true")
    parser.add_argument("-l", "--load", help='Ne calcule pas de matrice et load une matrice déjà faite', type=str)
    parser.add_argument("-s", "--save", help='Sauvegarde la matrice construite', type=str)

    args = parser.parse_args()

    args.fantom = True

def main():

  global args
  
  if args.load:
    DATA = np.load(args.load)
  else:
    DATA = construire_matrice_base(args.images_path,20)
    if args.save:
        np.save(args.save, DATA)

  if args.process:
    b = process_matrix(DATA)
    valp,espp = b[0],b[1]
    print('DATA :', DATA.shape)
    print('valeurs propres :', valp.shape)
    print('vecteurs propres :', espp.shape)
    data_reduced = reduction(DATA)
    print('data_reduced :', data_reduced.shape)
    print(np.sum(valp))
    print(valp)


if __name__ == "__main__":
    init_args()
    main()
    
