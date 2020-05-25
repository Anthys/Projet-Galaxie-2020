import libs.matrices3 as mt

def main():
  a = open("osef/truc.txt")
  lib = []
  folder = "JAIPASLESIMAGES"
  for line in a:
    lib.append(line.replace("\n", "")+".fits")
  mt.show_images_from_names(lib, folder, 3)