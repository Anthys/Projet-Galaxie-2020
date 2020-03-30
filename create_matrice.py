import os,sys,argparse
import numpy as np
from libs.pic_process import *
from libs.minkos import *
from libs.matrices import *

import matplotlib.pyplot as plt
import matplotlib

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("dat_path", help='', type=str)
    parser.add_argument("conv_path", help='', type=str)
    parser.add_argument("-p", "--process", help='',action="store_true")
    parser.add_argument("-n", "--no_treat", help='Pas de traitement et renomage des fichiers (si vous l avez deja fait)',action="store_true")
    parser.add_argument("-l", "--load", help='Ne calcule pas la matrice et load le fichier deja fait',action="store_true")

    args = parser.parse_args()

    args.fantom = True

def main():
  global args
  
  if not args.no_treat:
    traiter_dat(args.conv_path)
    traiter_dat(args.dat_path)
  
  if args.load:
    a = np.load("out.npy")
  else:
    a = calculer_matrice_base(args.dat_path, args.conv_path,50)
    np.save("out", a)

  if args.process:
    a = a.T
    example_matrix = np.array([
      [1,1,1],
      [1,1,5],
      [2,2,3],
      [3,3,1],
      [3,3,5]
      ], dtype=float)
    #a = example_matrix
    print(a.shape)
    print(a.T[0])
    b = process_matrix(a)
    #cercle_correlation(b[1], b[0])
    #sphere_correlation(b[1],b[0])
    #histograme_valeurs_propres(b[0], 10)
    #val_prop_espace(b[0])
    #print(b)
    global_curve(a)
  #print(b)


if __name__ == "__main__":
    init_args()
    main()
    
