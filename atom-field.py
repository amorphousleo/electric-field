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

inner_cutoff_radius = 1.0
outer_cutoff_radius = 3.0

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

def kdtree(structure, inner_cutoff_radius=1.0, outer_cutoff_radius=3.0):
    supercell_structure = structure * (3,3,3)
    offset = np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c])
    supercell_coordinates = supercell_structure.cart_coords - offset #np array of all coordinates
    coordinates = structure.cart_coords
    charges = np.array([getattr(site.specie, "oxi_state", 0) for site in supercell_structure])
    
    # Now build the KTTree
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(supercell_coordinates)
    atom_indicies = kdtree.query_ball_point(coordinates, outer_cutoff_radius)
    grid_efield = calculate_electric_field(coordinates,supercell_coordinates,inner_cutoff_radius, atom_indicies, charges)
    return grid_efield
    
@numba.jit
def calculate_electric_field(coordinates, supercell_coordinates,inner_cutoff_radius, atom_indicies, charges ):
    grid_efield = np.zeros([len(coordinates),3])
    
    n = 0
    for i in coordinates:
        for j in atom_indicies[n]:
            dist = np.linalg.norm(coordinates[n]-supercell_coordinates[j])
            if dist > 1.0:
                grid_efield[n] += ((coordinates[n]-supercell_coordinates[j])*((COLUMB * E2 * charges[n]) / (dist)))
        n += 1
    return grid_efield
        

if __name__ == "__main__":
    input_filename = "data/input/sep23.1_final_structure_new"
    a, b, c = 15.341, 15.341, 15.341

    # Construct Structure
    lattice = pmg.Lattice.from_parameters(a, b, c, 90, 90, 90)
    charges = {'Pt': 10, 'Ni': 10, 'P': 5}
    structure = read_structure(input_filename, lattice, charges)

    grid_efield = kdtree(structure, inner_cutoff_radius, outer_cutoff_radius)

    np.save('data/results/grid_efield.npy', grid_efield)
