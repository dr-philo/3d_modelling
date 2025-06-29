# helper_functions.py
import numpy as np
import io

# Covalent radii in Angstroms for common elements
# Source: Cambridge Structural Database and other standard sources.
COVALENT_RADII = {
    'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33,
    # Add other elements as needed
}

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
    """
    Writes atomic symbols and coordinates to an XYZ format string.
    """
    output = io.StringIO()
    output.write(f"{len(atomic_symbols)}\n")
    output.write("Modified molecule\n")
    for symbol, (x, y, z) in zip(atomic_symbols, atomic_coordinates):
        output.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    return output.getvalue()


def create_xyz_string(atomic_symbols, atomic_coordinates):
    """
    Creates an XYZ format string for visualization.
    """
    xyz_string = f"{len(atomic_symbols)}\nModified molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string


def _get_neighbors(atomic_symbols, atomic_coordinates, atom_index):
    """
    Identifies the bonded neighbors of a given atom based on covalent radii.
    """
    neighbors = []
    base_coords = atomic_coordinates[atom_index]
    base_symbol = atomic_symbols[atom_index]

    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        if i == atom_index:
            continue
        
        # Calculate expected bond length
        expected_bond_length = COVALENT_RADII.get(base_symbol, 1.0) + COVALENT_RADII.get(symbol, 1.0)
        # Check if the distance is within a tolerant range of the expected bond length (e.g., 1.2 times)
        distance = np.linalg.norm(base_coords - coords)
        if distance < expected_bond_length * 1.2:
            neighbors.append(i)
    return neighbors


def _calculate_new_atom_position(base_atom_coords, neighbor_coords_list, bond_length):
    """
    Calculates the position for a new atom based on hybridization rules.
    - 0 neighbors: Places the new atom along the x-axis.
    - 1 neighbor: Places the new atom opposite the neighbor (sp, linear).
    - 2 neighbors: Places the new atom to form a trigonal planar geometry (sp2).
    - 3+ neighbors: Places the new atom to form a tetrahedral geometry (sp3).
    """
    if not neighbor_coords_list:
        # No neighbors, place along an arbitrary vector (e.g., x-axis)
        direction_vector = np.array([1.0, 0.0, 0.0])
    else:
        # Sum of vectors from the base atom to its neighbors
        sum_of_vectors = np.sum([np.array(nc) - base_atom_coords for nc in neighbor_coords_list], axis=0)
        
        # The direction for the new bond is opposite to the sum of existing bond vectors
        direction_vector = -sum_of_vectors

        # Normalize the direction vector
        norm = np.linalg.norm(direction_vector)
        if norm < 1e-6: # Avoid division by zero if vectors cancel out
            if len(neighbor_coords_list) == 1:
                 # This can happen if the only neighbor is at the origin and so is the base atom.
                 # A more robust fallback would be to find a perpendicular vector.
                 # For simplicity, we'll use a non-collinear vector.
                 temp_vec = np.array([1.0, 0.0, 0.0])
                 if np.linalg.norm(np.cross(sum_of_vectors, temp_vec)) < 1e-6:
                     temp_vec = np.array([0.0, 1.0, 0.0])
                 direction_vector = np.cross(sum_of_vectors, temp_vec)
            else: # Fallback for symmetric cases (e.g. linear CO2)
                direction_vector = np.array([1.0, 0.0, 0.0]) # default to x-axis
        
        direction_vector /= np.linalg.norm(direction_vector)

    return base_atom_coords + direction_vector * bond_length


def replace_atom_with_group(
    atomic_symbols, atomic_coordinates, atom_index, new_group
):
    """Replaces an atom with a new functional group, respecting local geometry."""
    
    removed_atom_coords = atomic_coordinates[atom_index]
    
    # Find neighbors *before* removing the atom
    neighbor_indices = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
    neighbor_coords = [atomic_coordinates[i] for i in neighbor_indices]
    
    # Remove the selected atom
    del atomic_symbols[atom_index]
    atomic_coordinates = np.delete(atomic_coordinates, atom_index, axis=0)
    
    # The first atom of the new group replaces the removed atom.
    # Its position is calculated based on the neighbors of the *original* atom.
    first_atom_symbol = new_group[0]
    bond_length = COVALENT_RADII.get(first_atom_symbol, 0.77) + COVALENT_RADII.get(atomic_symbols[neighbor_indices[0]], 0.77) if neighbor_indices else 1.54

    # The new group is placed relative to the 'ghost' position of the removed atom
    # and its neighbors.
    if not neighbor_coords:
        raise ValueError("Cannot replace an atom with no bonded neighbors.")

    # We use the average position of neighbors as a reference point for direction
    avg_neighbor_pos = np.mean(neighbor_coords, axis=0)
    direction_vector = removed_atom_coords - avg_neighbor_pos
    direction_vector /= np.linalg.norm(direction_vector)

    # Add new group atoms
    for i, new_atom_symbol in enumerate(new_group):
        # A simple linear placement for subsequent atoms in the group.
        # A more advanced implementation would require group templates.
        new_coords = removed_atom_coords + i * direction_vector * 1.0 # using a generic 1.0 A for intra-group spacing
        atomic_symbols.insert(atom_index + i, new_atom_symbol)
        atomic_coordinates = np.insert(atomic_coordinates, atom_index + i, new_coords, axis=0)
        
    return atomic_symbols, atomic_coordinates


def add_group_to_atom(
    atomic_symbols, atomic_coordinates, atom_index, new_group
):
    """Adds a functional group to a specific atom, respecting hybridization rules."""
    
    base_atom_coords = atomic_coordinates[atom_index]
    base_atom_symbol = atomic_symbols[atom_index]
    
    # Find neighbors of the base atom
    neighbor_indices = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
    neighbor_coords_list = [atomic_coordinates[i] for i in neighbor_indices]

    # Add new group atoms one by one
    current_base_coords = base_atom_coords
    current_base_symbol = base_atom_symbol
    
    for i, new_atom_symbol in enumerate(new_group):
        bond_length = COVALENT_RADII.get(current_base_symbol, 0.77) + COVALENT_RADII.get(new_atom_symbol, 0.77)
        
        # The first atom is placed based on the original atom's neighbors.
        # Subsequent atoms are placed linearly relative to the previously added atom.
        new_coords = _calculate_new_atom_position(current_base_coords, neighbor_coords_list, bond_length)
        
        atomic_symbols.append(new_atom_symbol)
        atomic_coordinates = np.vstack([atomic_coordinates, new_coords])
        
        # Update for the next atom in the chain
        neighbor_coords_list = [current_base_coords] 
        current_base_coords = new_coords
        current_base_symbol = new_atom_symbol

    return atomic_symbols, atomic_coordinates


def delete_atoms(atomic_symbols, atomic_coordinates, atom_indices):
    """Deletes atoms at the specified indices."""
    # Sort indices in reverse order to avoid shifting problems
    sorted_indices = sorted(atom_indices, reverse=True)
    
    for index in sorted_indices:
        del atomic_symbols[index]
        atomic_coordinates = np.delete(atomic_coordinates, index, axis=0)
        
    return atomic_symbols, atomic_coordinates
