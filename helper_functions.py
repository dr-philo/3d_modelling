# helper_functions.py
import numpy as np
import io

COVALENT_RADII = {
    'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33,
}

# (read_xyz, write_xyz, create_xyz_string, and _get_neighbors functions are unchanged from previous version)

def read_xyz(file):
    content = file.getvalue().decode("utf-8")
    lines = content.splitlines()
    if len(lines) < 2: raise ValueError("Invalid XYZ: insufficient lines")
    try: num_atoms = int(lines[0].strip())
    except ValueError: raise ValueError("Invalid XYZ: first line must be atom count")
    if len(lines) < num_atoms + 2: raise ValueError("Invalid XYZ: insufficient atom data")
    atomic_symbols, atomic_coordinates = [], []
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) < 4: raise ValueError(f"Invalid XYZ: line {i+1} is malformed")
        atomic_symbols.append(parts[0])
        atomic_coordinates.append(list(map(float, parts[1:4])))
    return atomic_symbols, np.array(atomic_coordinates)

def write_xyz(atomic_symbols, atomic_coordinates):
    output = io.StringIO()
    output.write(f"{len(atomic_symbols)}\nModified molecule\n")
    for symbol, (x, y, z) in zip(atomic_symbols, atomic_coordinates):
        output.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    return output.getvalue()

def create_xyz_string(atomic_symbols, atomic_coordinates):
    xyz_string = f"{len(atomic_symbols)}\nModified molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string

def _get_neighbors(atomic_symbols, atomic_coordinates, atom_index):
    neighbors, base_coords, base_symbol = [], atomic_coordinates[atom_index], atomic_symbols[atom_index]
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        if i == atom_index: continue
        expected_bond_length = COVALENT_RADII.get(base_symbol, 1.0) + COVALENT_RADII.get(symbol, 1.0)
        if np.linalg.norm(base_coords - coords) < expected_bond_length * 1.2:
            neighbors.append(i)
    return neighbors

def _rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2. """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8: return np.identity(3) if c > 0 else -np.identity(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def _calculate_attachment_vector(base_atom_coords, neighbor_coords_list):
    """ Calculates the ideal direction vector for attaching a new group. """
    if not neighbor_coords_list: return np.array([1.0, 0.0, 0.0])
    sum_of_vectors = np.sum([np.array(nc) - base_atom_coords for nc in neighbor_coords_list], axis=0)
    direction_vector = -sum_of_vectors
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

def _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, mode):
    """ Core function to handle both additions and substitutions using 3D templates. """
    # Handle simple cases (single atoms)
    if 'coords' not in group_template:
        group_template = {
            "symbols": group_template["symbols"],
            "coords": np.array([[0.0, 0.0, 0.0]]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        }

    base_atom_coords = atomic_coordinates[atom_index]
    base_atom_symbol = atomic_symbols[atom_index]
    
    # Determine the target bond vector
    neighbor_indices = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
    neighbor_coords = [atomic_coordinates[i] for i in neighbor_indices]
    target_bond_vector = _calculate_attachment_vector(base_atom_coords, neighbor_coords)
    
    # Get group template info
    gt_symbols = group_template["symbols"]
    gt_coords = group_template["coords"]
    gt_anchor_idx = group_template["anchor_index"]
    gt_attach_vec = group_template["attachment_vector"]

    # Calculate bond length for the new bond
    bond_length = COVALENT_RADII.get(base_atom_symbol, 0.77) + COVALENT_RADII.get(gt_symbols[gt_anchor_idx], 0.77)

    # Calculate rotation matrix to align the group template with the target vector
    rotation_matrix = _rotation_matrix_from_vectors(gt_attach_vec, target_bond_vector)
    rotated_gt_coords = gt_coords @ rotation_matrix.T

    # Calculate position of the new group's anchor atom
    if mode == 'add':
        anchor_position = base_atom_coords + target_bond_vector * bond_length
    else: # 'replace'
        anchor_position = base_atom_coords # The new anchor sits where the old atom was

    # Translate the group to its final position
    final_coords = rotated_gt_coords - rotated_gt_coords[gt_anchor_idx] + anchor_position

    # Modify the molecule's lists
    if mode == 'replace':
        # Delete the atom to be replaced
        atomic_symbols.pop(atom_index)
        atomic_coordinates = np.delete(atomic_coordinates, atom_index, axis=0)
        # Insert the new group atoms
        for i in range(len(gt_symbols)):
            atomic_symbols.insert(atom_index + i, gt_symbols[i])
            atomic_coordinates = np.insert(atomic_coordinates, atom_index + i, final_coords[i], axis=0)
    else: # 'add'
        for i in range(len(gt_symbols)):
            atomic_symbols.append(gt_symbols[i])
            atomic_coordinates = np.vstack([atomic_coordinates, final_coords[i]])

    return atomic_symbols, atomic_coordinates

def add_group_to_atom(atomic_symbols, atomic_coordinates, atom_index, group_template):
    return _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, 'add')

def replace_atom_with_group(atomic_symbols, atomic_coordinates, atom_index, group_template):
    return _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, 'replace')

def delete_atoms(atomic_symbols, atomic_coordinates, atom_indices):
    """Deletes atoms at the specified indices."""
    sorted_indices = sorted(atom_indices, reverse=True)
    for index in sorted_indices:
        if 0 <= index < len(atomic_symbols):
            del atomic_symbols[index]
            atomic_coordinates = np.delete(atomic_coordinates, index, axis=0)
    return atomic_symbols, atomic_coordinates
