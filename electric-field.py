""" General Notes
 - code looks like c code
 - use functions to group ideas
 - use existing packages for tedious tasks (especially ones where performance does not matter)
 - better variables names means less need for comments
 - algorithm > replace with numpy, numba, cython > make code more complex > parallize

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

def calculate_electric_field(structure, inner_cutoff=1.0, outer_cutoff=3.0, grid=(80, 80, 80)):
    x_p = np.linspace(0, structure.lattice.a, grid[0])
    y_p = np.linspace(0, structure.lattice.b, grid[0])
    z_p = np.linspace(0, structure.lattice.c, grid[0])
    grid_points = np.vstack(np.meshgrid(x_p,y_p,z_p)).reshape(3,-1).T

    supercell_structure = structure * (3, 3, 3)
    np.save('data/results/supercell_structure.npy', supercell_structure.cart_coords)
    np.save('data/results/structure.npy', structure.cart_coords)
    offset = np.array([structure.lattice.a, structure.lattice.b, structure.lattice.c])
    centered_coordinates = supercell_structure.cart_coords  - offset # np array of all coordinates
    np.save('data/results/centered_coordinates.npy', centered_coordinates)
    charges = np.array([getattr(site.specie, "oxi_state", 0) for site in supercell_structure])

    from scipy.spatial import cKDTree
    kdtree = cKDTree(centered_coordinates)
    grid_indicies = kdtree.query_ball_point(grid_points, outer_cutoff_radius)

    atom_efield = _calculate_electric_field(centered_coordinates, inner_cutoff_radius, grid_indicies, charges)
    return atom_efield, grid_points


@numba.jit
def _calculate_electric_field(centered_coordinates, inner_cutoff_radius, grid_indicies, grid_points, charges):
    atom_efield = np.zeros(len(grid_points))
    print(len(structure.cart_coords))
    for i, (indicies, grid_point) in enumerate(zip(grid_indicies, grid_points)):
        for index in indicies:
            dist = np.linalg.norm(grid_point - centered_coordinates[index])
            if dist > inner_cutoff_radius and outer_cutoff_radius > dist:
                atom_efield[i] += COLUMB * E2 * charges[index] / (dist*dist)                
    return atom_efield


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

    grid_size = (80, 80, 80)
    atom_efield, grid_points = calculate_electric_field(structure, inner_cutoff_radius, outer_cutoff_radius, grid=grid_size)

    np.save('data/results/atom_efield.npy', atom_efield)
    np.save('data/results/grid_points.npy', grid_points)
