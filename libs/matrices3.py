import numpy as np
import os,sys
from libs.minkos import *
from libs.pic_process import *
import matplotlib.pyplot as plt


def traiter_dat(path):
  """ Remplace les caractères -, \\x1d, + par d'autres caractères dans les noms des fichiers de path """
  for i in os.listdir(path):
    if i[-3:] in ["txt","dat"]:
      n_name = i.replace("-", "_").replace("\x1d", "").replace("+", "_")
      print(n_name)
      os.rename(path + i,path+ n_name)


def build_data_matrix(images_path, max_iter=100):
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


def sort_eigenvalues(valeursPropres):
  """ Rend les valeurs propres triées par ordre décroissant. 
  - Entrée : tableau des valeurs propres (format numpy)
  - Sortie : tableau contenant les 3-tuples (val, pourcentage, indice de l'espace propre correspondant)"""
  
  valeursPropres = valeursPropres.real
  for i in range(len(valeursPropres)):
    if np.isclose(0, valeursPropres[i]):
      valeursPropres[i] = 0
    assert valeursPropres[i] >= 0
  p = sum(valeursPropres)
  supertuples = [(valeursPropres[i], valeursPropres[i]/p,i) for i in range(len(valeursPropres))]
  supertuples.sort(reverse=True)

  return supertuples


def eigenvalues_histogram(valeursPropres, n):
  """ Affiche l'histogramme des n valeurs propres les plus importantes de valeursPropres 
  - Entrée : tableau des valeurs propres (format numpy)""" 
  assert n <= len(valeursPropres)

  valeursPropres = sort_eigenvalues(valeursPropres)

  fig = plt.figure()
  ax = fig.add_axes([0.1,0.1,0.8,0.8])
  valeursPropres = valeursPropres[:n] # Tronquer
  x,y = [],[]
  for j in range(len(valeursPropres)):
    val = valeursPropres[j]
    x += [j]
    y += [val[0]]
  ax.bar(x,y)
  plt.title("Éboulis des valeurs propres")
  plt.xlabel(r"Indice $i$")
  plt.ylabel(r"Valeur propre $\delta_i$")
  plt.show()


def compute_new_data_matrix(DATA, espp, valeursPropres, n, display2d=False, display3d=False):
  """ Calcule la nouvelle matrice de données évaluant chaque individu selon les nouvelles variables.\n
  Si n < len(valeursPropres), la nouvelle matrice comporte uniquement les n variables les plus dispersives.
  Si display2d est True, affiche la projection des individus dans le plan des 2 variables d'inertie maximale.
  Si display3d est True, affiche la projection des individus dans l'espace des 3 variables d'inertie maximale.
  - Entrée : matrice de données initiale, tableau des espaces propres, tableau des valeurs propres (format numpy), nombre de variables voulues
  - Sortie : nouvelle matrice de données, éventuellement projetée sur un nombre restreint de variables""" 

  assert n <= len(valeursPropres)

  valeursPropres = sort_eigenvalues(valeursPropres)
  valeursPropres = valeursPropres[:n]
  
  new_DATA = np.dot(reduction(DATA),espp)   
  indexes = []
  for v in valeursPropres:
    indexes.append(v[2])
  new_DATA = new_DATA[:, indexes]

  if display2d:
    size_window = [5,5]
    fig = plt.figure(figsize = (*size_window,))
    for i, indiv in enumerate(new_DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      plt.scatter(x1,y1, c='red')
      plt.grid()
      plt.title('Projections de chaque individu sur les 2\n premières composantes principales')
      plt.xlabel(r"Projection sur $X'_1$ (en unité de $\sigma'_1$)")
      plt.ylabel(r"Projection sur $X'_2$ (en unité de $\sigma'_2$)")

    plt.show()

  if not display2d and display3d:
    size_window = [5, 5]
    fig = plt.figure(figsize = (*size_window,))
    ax = fig.add_subplot(111, projection='3d')
    for i, indiv in enumerate(new_DATA):
      x1 = indiv[0]
      y1 = indiv[1]
      z1 = indiv[2]
      ax.scatter(x1,y1,z1, c='red')
      ax.set_xlabel(r"$Projection sur X'_1 (en unité de \sigma'_1)$")
      ax.set_ylabel(r"$Projection sur X'_2 (en unité de \sigma'_2)$")
      ax.set_zlabel(r"$Projection sur X'_3 (en unité de \sigma'_3)$")
      plt.title('Projections de chaque individu sur les 3\n premières composantes principales')

    plt.show()

  return new_DATA


