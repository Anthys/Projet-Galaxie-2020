import os, sys


txtFirst = ""
txtEnd = ""
count = 0
for i in os.listdir():
  if i[-3:] == "dat":
    count += 1
    txtFirst+="![alt text][Test"+ str(count)+ "]\n"
    txtEnd += "[Test" + str(count)+']: img/' + i[:-4] +'.png "SuperLesFonctionelles"\n'

a = open("all_res.md", "w+")
a.write(txtFirst)
a.write("\n")
a.write(txtEnd)
a.close()