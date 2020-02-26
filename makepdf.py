from fpdf import FPDF
import os, sys

pdf = FPDF()
# imagelist is the list with all image filenames

path = 'img/main_results_0/smooth_2/'
imagelist = os.listdir(path)
print(imagelist)

for image in imagelist:
    pdf.add_page()
    pdf.image(path + image, y= 100, w=170, h=140)
pdf.output("smooth_2.pdf", "F")