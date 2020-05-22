import os,sys,argparse
import numpy as np
import scipy.linalg

from libs.pic_process import *
from libs.minkos import *
from libs.matrices3 import *
import libs.matrices as mt

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
    replace_special_characters(args.images_path)
    DATA = build_data_matrix(args.images_path,800)
    if args.save:
        np.save(args.save, DATA)

  if args.process:   
    print("DATA is real :", np.all(DATA == np.real(DATA)))
    data_reduced = reduction(DATA)
    valp, espp = process_matrix(DATA)
    sorted_valp = sort_eigenvalues(valp)
    print('shape DATA :', DATA.shape)
    print('shape data_reduced :', data_reduced.shape)
    print('shape valeurs propres :', valp.shape)
    print('shape vecteurs propres :', espp.shape)
    print('somme des vp :', np.sum(valp), "pourcentage des 2 premieres :", sorted_valp[0][1] + sorted_valp[1][1])
    # print('tableau des vp :', valp)
    #eigenvalues_histogram(valp, 5)
    new_DATA = compute_new_data_matrix(DATA, espp, valp, 5)
    polygon = [(-20,10),(10,10),(10,0),(-20,0)]
    plot_DATA_3D_in_2D(new_DATA)#,polygon)
    print('shape new_DATA :', new_DATA.shape)


if __name__ == "__main__":
    init_args()
    main()
    
