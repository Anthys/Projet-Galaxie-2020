import argparse,sys,os
import matplotlib.pyplot as plt
import numpy as np

lib_path = os.path.abspath(os.path.join(__file__, '..',"..", 'libs'))
sys.path.append(lib_path)

from pic_process import *
from minkos import *

parser = ""
args = ""


def get_right(F,U,chi,arg):
    """
        Retourne la fonctionelle correspondant à l'argument et sa couleur
    """
    x = 0
    if arg == "f":
        x = F
    elif arg == "u":
        x = U
    elif arg == "chi":
        x = chi
    return x, func_col(arg)

def main(myFile):

  max_lin = args.max
  size_window = [10,8]
  fig = plt.figure(figsize = (*size_window,))

  beta = fig.add_subplot(111)

  x = np.linspace(0.0, max_lin, 100)

  
  file1,name,ext = get_image(myFile)
  print("Processing", name, "...")

  if args.smooth:
    file1 = smooth_file(file1, args.smooth)
  
  if args.contrastLinear:
      file1 = contrastLinear(file1, args.contrastLinear)
  
  F, U, Chi = calcul_fonctionelles(file1, max_lin)

  h,col = get_right(F,U,Chi, args.functional)
  h = h/coef_normalization_functional(h)
  xf,ff = normalize_on_x(x,h)

  print(len(xf), len(x))
  #beta.plot(x,h/np.max(h))
  beta.plot(xf,ff, linewidth = 4)
  #beta.plot(xf,ff, linewidth = 4)
  beta.plot(x,h)

  
  plt.show()
      
    
  
    

def init_args():

    global parser, args

    parser = argparse.ArgumentParser(description='TakeMeOn')
    parser.add_argument("file", help='file', type=str)
    parser.add_argument("-s", "--save", help="save at the specified path (no showing)", type=str)
    parser.add_argument("-cL", "--contrastLinear", help="multiply contrast by x", type = int, default=40)
    parser.add_argument("-m", dest="max", help="maximum of the linear space", type = int, default=40)
    parser.add_argument("-dat", "--dat", action="store_true", help="file is in dat format")
    parser.add_argument("-smooth", "--smooth", type = int, help="smooth", default = 0)
    parser.add_argument("-n", "--name", type = str, help="name of file")
    parser.add_argument("-f", "--functional", type = str, help="name of functional to show",choices=['f', 'u', 'chi'], default = "f")
    parser.add_argument("-nonorm", "--nonorm", action="store_true",help="No normalisation")
    parser.add_argument("-maxfiles", "--maxfiles", type = int,help="max_nb of files to process, 0 for infnity", default=3)
    args = parser.parse_args()

    args.drawall = True

if __name__ == "__main__":
    init_args()
    main(args.file)
