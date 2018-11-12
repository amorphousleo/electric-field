# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:31:45 2018

@author: Leo-Desktop
"""
import numpy as np

fname = "data/input/sep23.1_final_structure"
fname_new = "data/input/sep23.1_final_structure_new"
with open(fname) as f:
    content = f.read().split()

atom,xpos,ypos,zpos,text = [],[],[],[],[]
alat_ang = 15.341

n = 0
while n < len(content):
    atom.append(content[n])
    xpos.append(alat_ang*float(content[n+1]))
    ypos.append(alat_ang*float(content[n+2]))
    zpos.append(alat_ang*float(content[n+3]))
    n += 4

n = 0
while n < 256:
    text.append((str(atom[n])+' '+str(xpos[n])+ ' '+str(ypos[n])+ ' '+str(zpos[n])))
    n += 1      

with open(fname_new, 'w') as f:
    for item in text:
        f.write("%s\n" % str(item))
        

            
    