import os, sys

for i in os.listdir("sims"):
  if i[-4:] == ".dat":
   os.system("python3.5 carlo_putain_tu_deconne.py "+"sims/"+i + " -m 255 -cL 0.000001 -n -smooth 0 -s sofianejetaime")