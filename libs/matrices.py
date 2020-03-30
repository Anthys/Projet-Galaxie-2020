import numpy as np
import os,sys
from libs.minkos import *
from libs.pic_process import *
import matplotlib.pyplot as plt

def traiter_dat(path):
  """ Remplace les caractères chiants des images """
  for i in os.listdir(path):
    if i[-3:] in ["txt","dat"]:
      n_name = i.replace("-", "_").replace("\x1d", "").replace("+", "_")
      print(n_name)
      os.rename(path + i,path+ n_name)


def calculer_matrice_base(dat_path, conventionelle_path, max_iter=100):
  """ Rends une matrice de données avec les galaxies en lignes et les paramètres en colonnes """ 

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
  final = final.T

  return final


def process_matrix(matrix_orig):
  """ Implémentation de la méthode tirée de 'Probabilités, statistiques et analyses multicritères', de Mathieu Rouaud  """

  matrix = np.copy(matrix_orig.T)
  for i in range(matrix.shape[0]):
    matrix[i] = matrix[i] - np.mean(matrix[i])
  
  for i in range(matrix.shape[0]):
    std = np.std(matrix[i])
    if std != 0:
      matrix[i] = matrix[i]/std
    else:
      print("std = 0")

  matrix2 = 1/matrix.shape[1]*np.dot(matrix, matrix.T)

  matrix2 = matrix2.T
  #print(matrix2.shape)
  val_et_espaces = np.linalg.eig(matrix2)

  return val_et_espaces

def global_curve(data):
  """ Représente la courbe de la moyenne des fonctions avec leurs écarts types """
  data = data.T
  final = []
  for parameter in data:
    final += [[np.mean(parameter), np.std(parameter)]]
  
  fig = plt.figure()
  x = [i for i in range(len(final))]
  print(final[0][0])
  means = [i[0] for i in final]
  stds = [i[1] for i in final]
  ax = plt.gca()
  ax.plot(x, means)
  ax.fill_between(x, [means[i]+stds[i] for i in range(len(x))], [means[i]-stds[i] for i in range(len(x))], facecolor='blue', alpha=0.5)
  plt.show()

def val_prop_espace(valeursPropres):
  """ Rends les valeurs propres triées de la plus informative à la moins informative, sous la forme (val, pourcentage, indice de l'espace propre) """
  valeursPropres = valeursPropres.real
  #p = sum(valeursPropres)
  for i in range(len(valeursPropres)):
    if np.isclose(0, valeursPropres[i]):
      valeursPropres[i] = 0
    assert valeursPropres[i] >= 0
  p = sum(valeursPropres)
  supertuple = [(valeursPropres[i], valeursPropres[i]/p,i) for i in range(len(valeursPropres))]
  supertuple.sort(reverse=True)
  ## FORMAT: (Valeurpropre, Pourcentage par rapport à la somme, espace propre associé)
  return supertuple


def histograme_valeurs_propres(valeursPropres, n):
  """ Histograme des n premières valeurs propres """ 
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
  """ Cercle de corrélation sur les deux valeurs propres les plus informatives """
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
  """ Sphère de corrélation sur les trois valeurs propres les plus informatives """

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
