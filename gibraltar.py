import os,sys,argparse, noise
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
    #replace_special_characters(args.images_path)
    things = []
    for i in range(25):
      things += [make_donut((100,100),i)]
      things += [make_triangle((100,100),i)]
    DATA = build_data_matrix3(things,1000)[0]
    print(DATA.shape)
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
    print('somme des vp :', np.sum(valp), "pourcentage des 2 premieres :", sorted_valp[0][1] + sorted_valp[1][1])
    # print('tableau des vp :', valp)
    #eigenvalues_histogram(valp, 5)
    new_DATA = compute_new_data_matrix(DATA, espp, valp, 5)
    #polygon = [(-20,10),(10,10),(10,0),(-20,0)]
    plot_DATA_2D(new_DATA)
    print('shape new_DATA :', new_DATA.shape)
    
def make_donut(shape = (100,100), b=1):
  world = np.zeros(shape)
  radius1 = 0.2
  radius2=0.4
  alpha = b+1
  for i in range(shape[0]):
      for j in range(shape[1]):
        ii = (i-shape[0]/2)/shape[0]
        jj = (j-shape[1]/2)/shape[1]
        angle = math.atan2(ii,jj)
        cos = math.cos
        sin = math.sin
        r2 = radius2+ noise.pnoise2(cos(angle),sin(angle),base=alpha)/10
        r1 = radius1+ noise.pnoise2(cos(angle),sin(angle),base=100+1+alpha)/10

        rho = math.sqrt(ii*ii+jj*jj)

        if rho < r2 and rho > r1:
          #world[i][j] = 1
          middle_rho = (r2+r1)/2
          max_dist = max(abs(r2-middle_rho), abs(r1-middle_rho))
          temp = 1-abs(rho-middle_rho)/max_dist
          if temp < 0:
            temp = 0
          world[i][j] = temp
  return world

def make_triangle(shape = (100,100),b=1):
  world = np.zeros(shape)
  radius1 = 0.2
  radius2=0.4
  alpha = b+1
  for i in range(shape[0]):
      for j in range(shape[1]):
        cos = math.cos
        sin = math.sin
        iii = (i-shape[0]/2)/shape[0]
        jjj = (j-shape[1]/2)/shape[1]
        theta = b*math.pi/6
        ii = cos(theta)*iii-sin(theta)*jjj
        jj = sin(theta)*iii + cos(theta)*jjj
        angle = math.atan2(ii,jj)
        r2 = radius2+ noise.pnoise2(cos(angle),sin(angle),base=alpha)/10
        r1 = radius1+ noise.pnoise2(cos(angle),sin(angle),base=100+1+alpha)/10

        rho = math.sqrt(ii*ii+jj*jj)

        a1,b1,a2,b2,a3,b3 = -0.2,-0.2,0.2,0.2, 1, 1

        if (True or ii>a1 +noise.pnoise1(jj, base = alpha)) and jj >b1+noise.pnoise1(ii*5, base = alpha)/5 and jj*jj<ii*ii and jj<b2-noise.pnoise1(ii*5, base = alpha+10)/5 :#and ii < a2-noise.pnoise1(ii):
          #world[i][j] = 1
          #middle_rho = (r2+r1)/2
          #max_dist = max(abs(r2-middle_rho), abs(r1-middle_rho))
          #temp = 1-abs(rho-middle_rho)/max_dist
          #if temp < 0:
          #  temp = 0
          maxdist = dist(a1,b1,(a1+0)/2,(b1+b2)/2)
          temp = 1-dist((a1+0)/2,(b1+b2)/2, ii,jj)/maxdist
          temp = max(temp, 1-dist((a2+0)/2,(b1+b2)/2, ii,jj)/maxdist)
          if temp <0:
            temp= 0
          world[i][j] = temp
  return world


def dist(x1,y1,x2,y2):
  return math.sqrt((x2-x1)**2+(y1-y2)**2)

if __name__ == "__main__":
    init_args()
    main()
