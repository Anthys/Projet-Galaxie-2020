import numpy as np
import os,sys
from libs.minkos import *
from libs.pic_process import *
import matplotlib.pyplot as plt


def construire_matrice_base(images_path, max_iter=100):
  """ Construit la matrice de données DATA contenant toutes les observations (MF sans ACS) de tous les individus
  - Entrée : chemin relatif vers le dossier contenant toutes les images
  - Sortie : matrice DATA au format n*p avec n le nombre d'individus et p le nombre de variables """ 

  initial = True

  images_list = os.listdir(images_path)

  for i,v in enumerate(images_list):
    ext = v[-4:]

    if i > max_iter:
      break
    if ext=="fits" or ext==".dat":
      print("Fichier trouvé, en cours de process")
      image_file = images_path + "/" + v

      data_fonctionnelles = get_image(image_file)
      data_fonctionnelles = contrastLinear(data_fonctionnelles[0], 70)

      F,U,Chi = calcul_fonctionelles(data_fonctionnelles, 256)
      F,U,Chi = np.array(F), np.array(U), np.array(Chi)
      N = np.hstack((F,U,Chi))

      if initial:
        DATA = N
        initial = False
      else:
        DATA = np.vstack((DATA, N))

  return DATA


def reduction(m):
  """Réduit la matrice m, c'est-à-dire :
  - Soustrait à chaque colonne sa moyenne
  - Divise chaque colonne par son écart-type.
  Puis renvoie une nouvelle matrice m' du même format que m."""

  matrix = np.copy(m.T)

  for i in range(matrix.shape[0]):
    matrix[i] = matrix[i] - np.mean(matrix[i])
  
  for i in range(matrix.shape[0]):
    std = np.std(matrix[i])
    if std != 0:
      matrix[i] = matrix[i]/std
    # else:
      # print("std = 0")
      
  return matrix.T


def process_matrix(DATA):
  """ Calcule les valeurs et vecteurs propres correspondant aux composantes principales de DATA.
  - Entrée : matrice initiale DATA au format n*p avec n le nombre d'individus et p le nombre de variables
  - Sortie : 2-tuple contenant la liste des valeurs propres et la liste des vecteurs propres
  Implémentation de la méthode tirée de 'Probabilités, statistiques et analyses multicritères', de Mathieu Rouaud.  """

  data_reduced = reduction(DATA)
  matrice_correlation = 1/data_reduced.shape[0] * np.dot(data_reduced.T, data_reduced)
  val_et_espaces = np.linalg.eig(matrice_correlation)

  return val_et_espaces     # pas forcément dans l'ordre souhaité

