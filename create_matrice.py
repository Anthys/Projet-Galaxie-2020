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
    # val_prop_espace(b[0])
    PCA(a, b[1], b[0])
    

def val_prop_espace(valeursPropres):
  valeursPropres = valeursPropres.real
  #p = sum(valeursPropres)
  for i in range(len(valeursPropres)):
    if np.isclose(0, valeursPropres[i]):
      valeursPropres[i] = 0
    assert valeursPropres[i] >= 0
  p = sum(valeursPropres)
  valeursPropres = [(valeursPropres[i], valeursPropres[i]/p,i) for i in range(len(valeursPropres))]
  valeursPropres.sort(reverse=True)
  ## FORMAT: (Valeurpropre, Pourcentage par rapport à la somme, espace propre associé)
  return valeursPropres


def histograme_valeurs_propres(valeursPropres, n):
  #print(valeursPropres)
  assert n < len(valeursPropres)
  valeursPropres = valeursPropres.real
  #p = sum(valeursPropres)
  valeursPropres = [(valeursPropres[i],i) for i in range(len(valeursPropres))]
  valeursPropres.sort(reverse=False)
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  valeursPropres = valeursPropres[:n] # Tronquer
  #print(valeursPropres[0][0], valeursPropres[3][0])
  x,y = [],[]
  for j in range(len(valeursPropres)):
    val = valeursPropres[j]
    x += [val[1]]
    y += [val[0]]
  ax.bar(x,y)
  plt.show()
    

def cercle_correlation(matriceEspaces, valeursPropres, n=2):
  assert n == 2
  matriceEspaces = matriceEspaces.T
  valeursPropres = [(valeursPropres[i],i) for i in range(len(valeursPropres))]
  valeursPropres.sort()
  valeursPropres = valeursPropres[:n]
  print(valeursPropres[0], valeursPropres[1])


  size_window = [5,5]
  fig = plt.figure(figsize = (*size_window,))
  #fig.add_subplot(111)
  for espace in matriceEspaces:
    x1 = espace[valeursPropres[0][1]]
    y1 = espace[valeursPropres[1][1]]
    plt.scatter(x1,y1)
  circ1 = plt.Circle((0,0), 1, fill= False, color='r')
  ax = plt.gcf().gca()
  val = 1.1
  ax.set_xlim([-val,val])
  ax.set_ylim([-val,val])
  plt.gcf().gca().add_artist(circ1)
  plt.show()


def sphere_correlation(matriceEspaces, valeursPropres, n=3):


  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_aspect("equal")

  u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
  x = np.cos(u)*np.sin(v)
  y = np.sin(u)*np.sin(v)
  z = np.cos(v)
  ax.plot_wireframe(x, y, z, color="r")

  ax.scatter([0], [0], [0], color="g", s=100)

  assert n == 3
  matriceEspaces = matriceEspaces.T
  matriceEspaces = matriceEspaces.real
  valeursPropres = [(valeursPropres[i],i) for i in range(len(valeursPropres))]
  valeursPropres.sort()
  valeursPropres = valeursPropres[:n]
  print(valeursPropres[0], valeursPropres[1], valeursPropres[2])

  #fig.add_subplot(111)
  for espace in matriceEspaces:
    x1 = espace[valeursPropres[0][1]]
    y1 = espace[valeursPropres[1][1]]
    z1 = espace[valeursPropres[2][1]]
    ax.scatter([x1],[y1],[z1], s=100)
  #circ1 = plt.Circle((0,0), 1, fill= False, color='r')
  #ax = plt.gcf().gca()
  #val = 1.1
  #ax.set_xlim([-val,val])
  #ax.set_ylim([-val,val])
  #plt.gcf().gca().add_artist(circ1)
  plt.show()


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
    
