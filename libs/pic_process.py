import numpy as np
import math

from astropy.io import fits
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import Gaussian2DKernel



def contrast(file1):
  """
    A ne pas utiliser pour le moment, on doit trouver une meilleure fonction mathématique
  """
  return file1
  """file1 = np.float64(file1)
  file1 = ((file1 - 128) / 128) * (np.pi / 2)
  file1 = 1*np.tanh(file1) + 1
  file1 = (file1 * 128) / 1
  return file1"""

def contrastLinear(file1, value):
  file1 = value*file1
  return file1


def fantom(file1):
  """
    Enlève les pixels fantomes créés par le crop de ds9
  """
  file2 = []
  for i in range(len(file1)):
      file3 = []
      for c in range(len(file1[i])):
          if not np.isnan(file1[i][c]):
              file3 += [file1[i][c]]
      if file3 != []:
          file2 += [file3]
  file2 = np.float64(file2)
  return file2

def smooth_file(file1, valsmooth):
  """
    Lisse une image avec une matrice gaussiene
  """
  size_gauss = valsmooth
  img = file1
  img_zerod = img.copy()
  # img_zerod[np.isnan(img)] = 0
  # It is a 9x9 array
  kernel = Gaussian2DKernel(size_gauss)#x_stddev=1)
  file1 = scipy_convolve(img, kernel, mode='same', method='direct')
  return file1

def get_dat_file(name):
  file1 = np.loadtxt(name)
  file1 = np.float64(file1)
  return file1

def get_fit_file(name):
  file1 = fits.getdata(name)
  file1 = np.float64(file1)
  return file1

def charger_le_putain_de_fichier(path):
  matrix = []
  file_r = open(path, "r")
  for line in file_r:
    temp = line.split()
    if temp[0] != "#":
        x = int(temp[2])-1
        y = int(temp[3])-1  
        v = float(temp[4])
        if y >= len(matrix):
          matrix.append([])
        if x >= len(matrix[y]):
          matrix[y].append([])
        if v == 0:
          pass
          #print("coucou")
        matrix[y][x] = v
  matrix = np.float64(matrix)
  name = path.split("/")[1].split(".")[0]
  return matrix, name

def get_image(path, dat=False):
  """
    Obtenir une image en format np-array-64, son nom et son extension
  """
  name = path.split("/")
  name = name[-1]
  ext = ""

  if "." in name:
    name = name.split(".")
    ext = name[-1]
    name = name[-2]


  # Récupérer le fichier
  if dat or ext == "dat":
    file1 = get_dat_file(path)
  elif ext == "fits":
    file1 = get_fit_file(path)
    # Enlever les pixels fantomes
    file1 = fantom(file1)
  return file1,name,ext

def degrade(file1, val):
    """ Dégrade la qualité d'une image en diminuant son nombre de pixels, val est le facteur de division """
    assert type(val) == int
    img1 = np.float64(file1)
    img2 = []
    for i in range(math.floor(img1.shape[0]/val)):
        img2 += [[]]
        for j in range(math.floor(img1.shape[1]/val)):
            moyenne = (img1[i*val][j*val] + img1[i*val + 1][j*val] + img1[i*val][j*val + 1] + img1[i*val + 1][j*val + 1])/(val**2)
            img2[i] += [moyenne]
    img2 = np.float64(img2)
    return img2