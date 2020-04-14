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

def calculer_matrice_base_sans_ACS(dat_path, max_iter=100):
  """ Rend une matrice de données avec les galaxies en lignes et les paramètres en colonnes """ 

  resolution = 100

  #matrice_fonctionelles = np.empty([resolution*2])
  initial = True

  dat_list = os.listdir(dat_path)

  for i,v in enumerate(dat_list):
    ext = v[-4:]
    print(ext)
    if i > max_iter:
      break
    if ext=="fits":
      print("Fichier trouvé, en cours de process")
      dat_file = dat_path + "/" + v

      data_fonctionelles = get_image(dat_file)
      data_fonctionelles = contrastLinear(data_fonctionelles[0], 70)
      F,U,Chi = calcul_fonctionelles(data_fonctionelles, 256)
      F,U,Chi = np.array(F), np.array(U), np.array(Chi)
      N = np.hstack((F,U,Chi))
      N = N.T
      #print(N.shape)
      #print(matrice_fonctionelles.shape )

      if initial:
        matrice_fonctionelles = N
        initial = False
      else:
        matrice_fonctionelles = np.vstack((matrice_fonctionelles, N))

  return matrice_fonctionelles

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("dat_path", help='', type=str)
    #parser.add_argument("conv_path", help='', type=str)
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
    a = calculer_matrice_base_sans_ACS(args.dat_path,50)
    np.save("out", a)

  if args.process:
    #a = a.T
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
    #a = example_matrix
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
    histograme_valeurs_propres(b[0], 10)
    #val_prop_espace(b[0]
    #print(b)
    #print(a.shape())
    #global_curve(a)
    #individus_en_nouvelle_base_tronquée(red, espp, valp, 2)
    print(PCA(a, b[1], b[0]))
    #print(b)


if __name__ == "__main__":
    init_args()
    main()
    
