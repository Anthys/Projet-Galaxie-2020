import libs.matrices3 as mt

def main():
  a = open("osef/truc4.txt")
  lib = []
  folder = "DONTPUSH"
  for line in a:
    lib.append(line.replace("\n", "").replace(' ', '')+".fits")
  mt.show_images_from_names(lib, folder, 5)

if __name__ == "__main__":
    main()