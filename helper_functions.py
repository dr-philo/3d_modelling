# helper_functions.py
import numpy as np
import io

def read_xyz(file):
    """
    Read molecular structure from XYZ file format.
    Handles blank comment lines properly.
    """
    content = file.getvalue().decode("utf-8")
    lines = content.splitlines()

    if len(lines) < 2:
        raise ValueError("Invalid XYZ file format: insufficient lines")

    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid XYZ file format: first line must be atom count")

    if len(lines) < num_atoms + 2:
        raise ValueError("Invalid XYZ file format: insufficient atom data")

    atomic_symbols = []
    atomic_coordinates = []

    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ file format: line {i+1} is malformed")

        symbol = parts[0]
        x, y, z = map(float, parts[1:4])
        atomic_symbols.append(symbol)
        atomic_coordinates.append([x, y, z])

    return atomic_symbols, np.array(atomic_coordinates)


def write_xyz(atomic_symbols, atomic_coordinates):
    output = io.StringIO()
    output.write(f"{len(atomic_symbols)}\n")
    output.write("Modified molecule\n")
    for symbol, (x, y, z) in zip(atomic_symbols, atomic_coordinates):
        output.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    return output.getvalue()


def create_xyz_string(atomic_symbols, atomic_coordinates):
    xyz_string = f"{len(atomic_symbols)}\nModified molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string


def replace_atom_with_group(
    atomic_symbols, atomic_coordinates, atom_index, new_group, bond_length=1.0
):
    # Remove the selected atom
    removed_atom_coords = atomic_coordinates[atom_index]
    del atomic_symbols[atom_index]
    atomic_coordinates = np.delete(atomic_coordinates, atom_index, axis=0)

    # Find the nearest neighbor (assuming it's bonded)
    distances = np.linalg.norm(atomic_coordinates - removed_atom_coords, axis=1)
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_coords = atomic_coordinates[nearest_neighbor_index]

    # Calculate the bond vector
    bond_vector = removed_atom_coords - nearest_neighbor_coords
    bond_vector /= np.linalg.norm(bond_vector)

    # Add new group atoms
    for i, new_atom in enumerate(new_group):
        new_coords = nearest_neighbor_coords + (i + 1) * bond_length * bond_vector
        atomic_symbols.append(new_atom)
        atomic_coordinates = np.vstack([atomic_coordinates, new_coords])

    return atomic_symbols, atomic_coordinates


def add_group_to_atom(
    atomic_symbols, atomic_coordinates, atom_index, new_group, bond_length=1.0
):
    base_atom_coords = atomic_coordinates[atom_index]

    # Find the nearest neighbor (assuming it's bonded)
    distances = np.linalg.norm(atomic_coordinates - base_atom_coords, axis=1)
    distances[atom_index] = np.inf  # Exclude the atom itself
    nearest_neighbor_index = np.argmin(distances)
    nearest_neighbor_coords = atomic_coordinates[nearest_neighbor_index]

    # Calculate the bond vector (pointing away from the nearest neighbor)
    bond_vector = base_atom_coords - nearest_neighbor_coords
    bond_vector /= np.linalg.norm(bond_vector)

    # Add new group atoms
    for i, new_atom in enumerate(new_group):
        new_coords = base_atom_coords + (i + 1) * bond_length * bond_vector
        atomic_symbols.append(new_atom)
        atomic_coordinates = np.vstack([atomic_coordinates, new_coords])

    return atomic_symbols, atomic_coordinates


def delete_atoms(atomic_symbols, atomic_coordinates, atom_indices):
    # Sort indices in reverse order to avoid shifting problems
    for index in sorted(atom_indices, reverse=True):
        del atomic_symbols[index]
        atomic_coordinates = np.delete(atomic_coordinates, index, axis=0)
    return atomic_symbols, atomic_coordinates
