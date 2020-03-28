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
    a = a.T #à enlever dès qu'on créera une nouvelle matrice sans -load
    b = process_matrix(a)
    #print(b)
    #cercle_correlation(b[1], b[0])
    #sphere_correlation(b[1],b[0])
    #histograme_valeurs_propres(b[0], 10)
    #val_prop_espace(b[0])
    global_curve(a)
  #print(b)


def PCA(a, matriceEspaces, valeursPropres, proportion=0.99999999):

  matriceEspaces = matriceEspaces.T
  valeursSorted = val_prop_espace(valeursPropres)

  vecteursPropres = []
  prop = 0
  i = 0

  while prop < proportion:
    vecteursPropres.append(matriceEspaces[:,valeursSorted[i][2]])
    prop += valeursSorted[i][1]
    i += 1

  nb_variables = len(vecteursPropres) # seulement les variables qui contiennent p*100 % de l'info
  nb_galaxies = np.shape(a)[0]

  data_standardized = a.T

  for i in range(nb_galaxies):
    data_standardized[i] = data_standardized[i] - np.mean(data_standardized[i])

  for i in range(nb_galaxies):
    std = np.std(data_standardized[i])
    if std != 0:
      data_standardized[i] = data_standardized[i]/std
  
  data_standardized = data_standardized.T

  result = np.zeros((nb_galaxies, nb_variables))
  for index_variable in range(nb_variables):
    for index_galaxie in range(nb_galaxies):
      X = data_standardized[index_galaxie, :]
      result[index_galaxie, index_variable] = np.dot(X, vecteursPropres[index_variable])

  variables = np.arange(nb_variables)
  for index_galaxie in range(nb_galaxies):
    plt.plot(variables, result[index_galaxie, :], 'o')
  plt.show()

  return result


if __name__ == "__main__":
    init_args()
    main()
    
