# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:58:10 2018

@author: Leo-Desktop
"""

import pymatgen as pmg
import numpy as np
import numba

# Constants Section 
# Coulumb Constant : Units : N m^2 C^-2
COLUMB = 9.0*10**29
E2 = (1.602*10**-19)**2


def read_structure(filename, lattice, charges):
    # Lattice was not specified in file
    # Charges supplied as dict { specie : charge }
    with open(filename) as f:
        species = []
        positions = []
        for line in f:
            specie, x, y, z = line.split()
            species.append(specie)
            positions.append([float(_) for _ in [x, y, z]])
    structure = pmg.Structure(lattice, species, positions, coords_are_cartesian=True)
    structure.add_oxidation_state_by_element(charges)
    return structure

def calculate_electric_field(structure, inner_cutoff=1.0, outer_cutoff=3.0):
    grid_

if __name__ == "__main__":
    input_filename = "data/input/sep23.1_final_structure_new"
    a, b, c = 15.341, 15.341, 15.341

    # Inner and outer cutoff define the range of the Columb potential
    # We have an inner cutoff to ignore self field
    inner_cutoff_radius = 1.0
    outer_cutoff_radius = 3.0  

    # Construct Structure
    lattice = pmg.Lattice.from_parameters(a, b, c, 90, 90, 90)
    charges = {'Pt': 10, 'Ni': 10, 'P': 5}
    structure = read_structure(input_filename, lattice, charges)

    grid_efield = calculate_electric_field(structure, inner_cutoff_radius, outer_cutoff_radius)

    np.save('data/results/grid_efield.npy', grid_efield)
