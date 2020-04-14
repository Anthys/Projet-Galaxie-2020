import os,sys,argparse
import numpy as np
from libs.pic_process import *
from libs.minkos import *
from libs.matrices import *

import matplotlib.pyplot as plt
import matplotlib
from copy import copy

from mpl_toolkits.mplot3d import Axes3D
from itertools import product, combinations

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("dat_path", help='', type=str)
    parser.add_argument("conv_path", help='', type=str)
    parser.add_argument("-p", "--process", help='',action="store_true")
    parser.add_argument("-n", "--no_treat", help="Pas de traitement et renomage des fichiers (si vous l'avez deja fait)",action="store_true")
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
    example_matrix2 = np.array([
    [55,25,14,16],
    [25,36,18,20],
    [90,25,30,15],
    [18,35,28,20],
    [18,42,34,20],
    [70,21,36,28],
    [25,32,41,20],
    [35,25,49,20],
    [150,28,50,13],
    [130,19,54,15],
    [35,26,106,20]
    ], dtype=float)
    example_matrix3 = np.array([
    [1350, 79, 393, 161, 870, 165],
    [1588, 85, 468, 177, 1110, 160],
    [1294, 68, 424, 168, 1050, 152],
    [1222, 59, 412, 161, 930, 151],
    [1585, 98, 439, 164, 1105, 165],
    [1297, 82, 429, 169, 1080, 160],
    [1796, 79, 449, 169, 1160, 154],
    [1565, 55, 424, 163, 1010, 140],
    [2664, 128, 452, 173, 1320, 180],
    [1166, 55, 399, 157, 815, 140],
    [1570, 109, 428, 162, 1060, 175],
    [1798, 82, 445, 172, 1160, 158],
    [1998, 115, 469, 169, 1370, 160],
    [1993, 98, 438, 170, 1080, 167],
    [1442, 80, 431, 166, 1129, 144],
    [1769, 83, 440, 165, 1095, 165],
    [1979, 100, 459, 173, 1120, 173],
    [1294, 68, 404, 161, 955, 140]
    ], dtype=float)
    a = example_matrix3
    b = process_matrix(a)
    valp,espp = b[0],b[1]
    print(valp)
    print(espp.shape)
    print(espp)
    print("---")
    red = reduction(a)
    print(red)
    print(np.dot(red,espp))
    print("--")
    #print(b)
    #rep_on_principal(b[1], b[0],a)
    #cercle_correlation(b[1], b[0],a)
    #sphere_correlation(b[1],b[0])
    histograme_valeurs_propres(b[0], 3)
    #val_prop_espace(b[0])
    #print(b)
    #global_curve(a)
    #print(PCA(a, b[1], b[0]))
    #print(b)


if __name__ == "__main__":
    init_args()
    main()
    
