import numpy as np
import os,sys
from libs.minkos import *
from libs.pic_process import *

def traiter_dat(path):
  for i in os.listdir(path):
    if i[-3:] in ["txt","dat"]:
      n_name = i.replace("-", "_").replace("\x1d", "").replace("+", "_")
      print(n_name)
      os.rename(path + i,path+ n_name)


def calculer_matrice_base(dat_path, conventionelle_path, max_iter=100):

  resolution = 100

  #matrice_fonctionelles = np.empty([resolution*2])
  initial = True
  ligne_C = []
  ligne_A = []
  ligne_S = []


  conventionelle_list = os.listdir(conventionelle_path)
  dat_list = os.listdir(dat_path)

  for i,v in enumerate(conventionelle_list):
    cur_name = v[:-8]
    if i > max_iter:
      break
    if cur_name + ".dat.txt" in conventionelle_list and cur_name + ".dat" in dat_list:
      print("Fichier trouvé, en cours de process")
      dat_file = dat_path + "/" + cur_name+".dat"

      data_fonctionelles = get_image(dat_file)
      data_fonctionelles = contrastLinear(data_fonctionelles[0], 70)
      F,U,Chi = calcul_fonctionelles(data_fonctionelles, 256)
      F,U = np.array(F), np.array(U)
      N = np.hstack((F,U))
      N = N.T
      #print(N.shape)
      #print(matrice_fonctionelles.shape )

      if initial:
        matrice_fonctionelles = N
        initial = False
      else:
        matrice_fonctionelles = np.vstack((matrice_fonctionelles, N))


      conv_file = open(conventionelle_path + "/" + cur_name + ".dat.txt")
      list_lines = conv_file.readlines()

      ligne_C += [float(list_lines[21][3:len(list_lines[21])-1])]
      ligne_A += [float(list_lines[22][3:len(list_lines[22])-1])]
      ligne_S += [float(list_lines[23][3:len(list_lines[23])-1])]

      conv_file.close()
      #print(matrice_fonctionelles[:,0])


  matrice_fonctionelles = matrice_fonctionelles.T
  ligne_S, ligne_C, ligne_A = np.array(ligne_S), np.array(ligne_C), np.array(ligne_A)

  final = np.vstack((matrice_fonctionelles,ligne_C,ligne_A,ligne_S))

  return final


def process_matrix(matrix):
  """ Process la matrice d'arès le protocole des notes sur Google Drive, rend les valeurs propres """

  for line in matrix:
    line = line - np.mean(line)

  for line in matrix:
    std = np.std(line)
    if std != 0:line = line/std

  matrix2 = 1/matrix.shape[0]*np.dot(matrix, matrix.T)

  val_propres = np.linalg.eig(matrix2)

  return val_propres