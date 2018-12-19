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
COLUMB = 8.987*10**19
E = 1.602*10**-19

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
    #structure.to(filename="data/input/NiPtP.xyz")
    structure.add_oxidation_state_by_element(charges)
    return structure

def kdtree(structure, inner_cutoff_radius=1.0, outer_cutoff_radius=3.0):
    supercell_structure = structure * (3,3,3)
    offset = np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c])
    supercell_coordinates = supercell_structure.cart_coords - offset #np array of all coordinates
    coordinates = structure.cart_coords
    coordinates_fractional = structure.frac_coords
    charges = np.array([getattr(site.specie, "oxi_state", 0) for site in supercell_structure])
    
    # Now build the KTTree
    
    from scipy.spatial import cKDTree
    kdtree = cKDTree(supercell_coordinates)
    atom_indicies = kdtree.query_ball_point(coordinates, outer_cutoff_radius)
    grid_efield = calculate_electric_field(coordinates,supercell_coordinates,inner_cutoff_radius, atom_indicies, charges)
    
    # Now output final csv file
    final_array = np.zeros([len(coordinates_fractional),6]) # Array with coordinates and efield values for each atom
    n = 0 
    for i in coordinates_fractional:
        final_array[n] = np.append(coordinates_fractional[n],grid_efield[n])
        n += 1
    
    e_filename = "D:/Github/electric-field/data/results/efield_sep18.1_cut3ang.csv"
    comment = "Columns: 1:(x-position), 2:(y-position), 3:(z-position), 4:(x-efield), 5:(y-efield), 6:(z-efield)| Cutoff:{} Angstroms | Units:(position-Angstroms, Electric Field-V/Angstrom)".format(outer_cutoff_radius)
    np.savetxt(e_filename,final_array,delimiter=',',header=comment)    
    return grid_efield
    
@numba.jit
def calculate_electric_field(coordinates, supercell_coordinates,inner_cutoff_radius, atom_indicies, charges ):
    grid_efield = np.zeros([len(coordinates),3])
    
    n = 0
    for i in coordinates:
        for j in atom_indicies[n]:
            dist = np.linalg.norm(coordinates[n]-supercell_coordinates[j])
            if dist > 1.0:
                grid_efield[n] += ((coordinates[n]-supercell_coordinates[j])*((COLUMB * E * charges[n]) / (dist**2)))
        n += 1
    return grid_efield
        

if __name__ == "__main__":
    input_filename = "data/input/sep18.1_final_structure_new"
    a, b, c = 15.3461, 15.3461, 15.3461

    # Construct Structure
    lattice = pmg.Lattice.from_parameters(a, b, c, 90, 90, 90)
    charges = {'Pt': 6, 'Ni': 2, 'P': 5}
    structure = read_structure(input_filename, lattice, charges)

    grid_efield = kdtree(structure, inner_cutoff_radius, outer_cutoff_radius)

    np.save('data/results/cut3_grid_efield.npy', grid_efield)
    np.save('data/results/coordinates.npy', structure.cart_coords) # save coordinates as np array

        
        
