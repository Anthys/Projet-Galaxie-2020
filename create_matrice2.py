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
    DATA = build_data_matrix(args.images_path,20)
    if args.save:
        np.save(args.save, DATA)

  if args.process:
    valp, espp = process_matrix(DATA)
    data_reduced = reduction(DATA)
    print('shape DATA :', DATA.shape)
    print('shape valeurs propres :', valp.shape)
    print('shape vecteurs propres :', espp.shape)
    print('shape data_reduced :', data_reduced.shape)
    print('somme des vp :', np.sum(valp))
    print('tableau des vp :', valp)
    new_DATA = compute_new_data_matrix(DATA, espp, valp, 4, display2d=True, display3d=False)
    print('shape new_DATA :', new_DATA.shape)
    for i in range(4):
      print('écart-type de la variable', i, ':', np.std(new_DATA[:,i]))




if __name__ == "__main__":
    init_args()
    main()
    
