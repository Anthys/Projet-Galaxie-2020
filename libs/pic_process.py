import numpy as np

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