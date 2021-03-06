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
    DATA = build_data_matrix(args.images_path,900)
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
    print('somme des vp :', np.sum(valp), "pourcentage des 3 premieres :", sorted_valp[0][1] + sorted_valp[1][1] + sorted_valp[2][1])
    #print('tableau des vp :', valp)
    #eigenvalues_histogram(valp, 10)
    new_DATA = compute_new_data_matrix(DATA, espp, valp, 8)
    print('shape new_DATA :', new_DATA.shape)
    #Nb_Cl = [i for i in range(2, 27)]
    #Inertia = []
    #for i in Nb_Cl:
      #Inertia.append(find_clusters(new_DATA, i)[1])
    #plt.plot(Nb_Cl, Inertia)
    #plt.xlabel("Nombre de clusters")
    #plt.ylabel("Inertie")
    #plt.show()


    labels, intertia = get_DATA_2D_in_clusters(new_DATA, 5)

    if True:
      separ_DATA = {}
      for i, indiv in enumerate(DATA):
        v = labels[i]
        if v not in separ_DATA.keys():
          separ_DATA[v] = []
        separ_DATA[v] += [indiv]
      for k,v in separ_DATA.items():
        separ_DATA[k] = np.float64(separ_DATA[k])
      cols = ["blue", "orange", "green", "red", "purple", "grey", "brown", "pink", "purple", "cyan", "beige", "deeppink"]
    
      mxi = min(5, len(cols), len(separ_DATA.keys()))
      size_window = [10,4]
      fig = plt.figure(figsize = (*size_window,))
      fig.add_subplot(1,4,2)
      for i,k in enumerate(separ_DATA.keys()):
        if i < mxi:
          mt.global_curve2(separ_DATA[k][:,:100], cols[i])
      plt.title("Aire F")
      fig.add_subplot(1,4,3)
      for i,k in enumerate(separ_DATA.keys()):
        if i < mxi:
          mt.global_curve2(separ_DATA[k][:,100:200], cols[i])
      plt.title("Périmètre U")
      fig.add_subplot(1,4,4)
      for i,k in enumerate(separ_DATA.keys()):
        if i < mxi:
          mt.global_curve2(separ_DATA[k][:,200:300], cols[i])
      plt.title(r"Caractéristique d'Euler $\chi$")

    fig.add_subplot(1,4,1)
    plot_DATA_2D_in_clusters(new_DATA, labels)
    plt.tight_layout( )
    plt.show()


if __name__ == "__main__":
    init_args()
    main()
    
