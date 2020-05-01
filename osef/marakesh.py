import noise, math
import numpy as np
import matplotlib.pyplot as plt

shape = (500,500)
scale = 100.0

les_images = []
n= 10
typee = "D"

txt = "Salut tout le monde c'est Squeezie j'espère que vous allez bien aujourd'hui petite vidéo gaming sur call of"
txt = txt.split()

fig, axs = plt.subplots(2,5, figsize=(10,8))
if typee == 'A':
  for f in range(n):
    world = np.zeros(shape)
    radius1 = 0.1
    radius2=0.2
    alpha = f+1
    for i in range(shape[0]):
        for j in range(shape[1]):
          ii = (i-shape[0]/2)/shape[0]
          jj = (j-shape[0]/2)/shape[1]
          if ii*ii+jj*jj < radius2+ noise.pnoise2(ii*4,jj*4,base=alpha)/10 and  ii*ii+jj*jj > radius1+ noise.pnoise2(ii*4,jj*4,base=alpha)/10:
            world[i][j] = 1
    axs[f//5,f%5].imshow(world)
elif typee == "B":
  for f in range(n):
    world = np.zeros(shape)
    radius1 = 0.1
    radius2=0.2
    alpha = f+1
    for i in range(shape[0]):
        for j in range(shape[1]):
          ii = (i-shape[0]/2)/shape[0]
          jj = (j-shape[0]/2)/shape[1]
          if ii*ii+jj*jj < radius2+ noise.pnoise1(math.atan2(jj,ii))/5 and  ii*ii+jj*jj > radius1:
            world[i][j] = 1
    axs[f//5,f%5].imshow(world)
elif typee == "C":
  for f in range(n):
    world = np.zeros(shape)
    radius1 = 0.1
    radius2=0.2
    alpha = f+1
    for i in range(shape[0]):
        for j in range(shape[1]):
          ii = (i-shape[0]/2)/shape[0]
          jj = (j-shape[0]/2)/shape[1]
          r2 = radius2+ noise.pnoise2(ii*4,jj*4,base=alpha)/10
          r1 = radius1+ noise.pnoise2(ii*4,jj*4,base=alpha)/10
          if ii*ii+jj*jj < r2 and  ii*ii+jj*jj > r1:
            world[i][j] = (abs( (math.sqrt(ii*ii+jj*jj))-(r2+r1)/2))
    axs[f//5,f%5].imshow(world)
elif typee == "D":
  for f in range(n):
    world = np.zeros(shape)
    radius1 = 0.2
    radius2=0.4
    alpha = f+1
    for i in range(shape[0]):
        for j in range(shape[1]):
          ii = (i-shape[0]/2)/shape[0]
          jj = (j-shape[1]/2)/shape[1]
          angle = math.atan2(ii,jj)
          cos = math.cos
          sin = math.sin
          r2 = radius2+ noise.pnoise2(cos(angle),sin(angle),base=alpha)/10
          r1 = radius1+ noise.pnoise2(cos(angle),sin(angle),base=n+1-alpha)/10

          rho = math.sqrt(ii*ii+jj*jj)

          if rho < r2 and rho > r1:
            #world[i][j] = 1
            middle_rho = (r2+r1)/2
            max_dist = max(abs(r2-middle_rho), abs(r1-middle_rho))
            temp = 1-abs(rho-middle_rho)/max_dist
            if temp < 0:
              temp = 0
            world[i][j] = temp
    axs[f//5,f%5].imshow(world)
    axs[f//5,f%5].set_title(txt[f])
        
plt.tight_layout()
plt.show()