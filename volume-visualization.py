# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:54:52 2018

@author: Leo-Desktop
"""

import pymatgen as pmg
import numpy as np

grid_efield = np.load("data/results/grid_efield.npy")
print(len(grid_efield))