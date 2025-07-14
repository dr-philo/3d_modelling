# helper_functions.py
import numpy as np
import io

# Covalent radii in Angstroms for common elements
COVALENT_RADII = {
    'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33,
}

def read_xyz(file):
    """
    Read molecular structure from XYZ file format.
    """
    content = file.getvalue().decode("utf-8")
    lines = content.splitlines()
    if len(lines) < 2: raise ValueError("Invalid XYZ: insufficient lines")
    try:
        num_atoms = int(lines[0].strip())
    except ValueError:
        raise ValueError("Invalid XYZ: first line must be atom count")
    if len(lines) < num_atoms + 2: raise ValueError("Invalid XYZ: insufficient atom data")
    
    atomic_symbols, atomic_coordinates = [], []
    for i in range(2, 2 + num_atoms):
        parts = lines[i].split()
        if len(parts) < 4: raise ValueError(f"Invalid XYZ: line {i+1} is malformed")
        atomic_symbols.append(parts[0])
        atomic_coordinates.append(list(map(float, parts[1:4])))
    return atomic_symbols, np.array(atomic_coordinates)

def write_xyz(atomic_symbols, atomic_coordinates):
    """
    Writes atomic symbols and coordinates to an XYZ format string.
    """
    output = io.StringIO()
    output.write(f"{len(atomic_symbols)}\nModified molecule\n")
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
    neighbors, base_coords, base_symbol = [], atomic_coordinates[atom_index], atomic_symbols[atom_index]
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        if i == atom_index: continue
        expected_bond_length = COVALENT_RADII.get(base_symbol, 1.0) + COVALENT_RADII.get(symbol, 1.0)
        if np.linalg.norm(base_coords - coords) < expected_bond_length * 1.2:
            neighbors.append(i)
    return neighbors

def _rotation_matrix_from_vectors(vec1, vec2):
    """
    Finds the rotation matrix that aligns vec1 to vec2.
    This version correctly handles the 180-degree case to avoid inversion.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.identity(3)

    if np.isclose(c, -1.0):
        p_axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(p_axis) < 1e-6:
            p_axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        p_axis /= np.linalg.norm(p_axis)
        kmat = np.array([[0, -p_axis[2], p_axis[1]], [p_axis[2], 0, -p_axis[0]], [-p_axis[1], p_axis[0], 0]])
        return np.identity(3) + 2 * kmat.dot(kmat)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def _calculate_attachment_vector(base_atom_coords, neighbor_coords_list):
    """
    Calculates the ideal direction vector for attaching a new group.
    """
    if not neighbor_coords_list:
        return np.array([1.0, 0.0, 0.0])
    sum_of_vectors = np.sum([np.array(nc) - base_atom_coords for nc in neighbor_coords_list], axis=0)
    direction_vector = -sum_of_vectors
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

def _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, mode):
    """ Core function to handle both additions and substitutions using 3D templates. """
    if 'coords' not in group_template:
        group_template = { "symbols": group_template["symbols"], "coords": np.array([[0.0, 0.0, 0.0]]), "anchor_index": 0, "attachment_vector": np.array([-1.0, 0.0, 0.0])}

    base_atom_coords = atomic_coordinates[atom_index]
    gt_symbols, gt_coords, gt_anchor_idx, gt_attach_vec = group_template["symbols"], group_template["coords"], group_template["anchor_index"], group_template["attachment_vector"]
    gt_anchor_symbol = gt_symbols[gt_anchor_idx]
    
    neighbor_indices = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
    neighbor_coords = [atomic_coordinates[i] for i in neighbor_indices]
    
    attach_to_atom_pos = None # Keep track of the atom we attach to
    if mode == 'add':
        target_bond_vector = _calculate_attachment_vector(base_atom_coords, neighbor_coords)
        bond_length = COVALENT_RADII.get(atomic_symbols[atom_index], 0.77) + COVALENT_RADII.get(gt_anchor_symbol, 0.77)
        anchor_position = base_atom_coords + target_bond_vector * bond_length
        attach_to_atom_pos = base_atom_coords
    else: # 'replace'
        if not neighbor_indices: raise ValueError("Cannot replace an atom with no bonded neighbors to attach to.")
        attach_to_atom_index = neighbor_indices[0]
        attach_to_atom_pos = atomic_coordinates[attach_to_atom_index]
        attach_to_atom_symbol = atomic_symbols[attach_to_atom_index]
        target_bond_vector = base_atom_coords - attach_to_atom_pos
        norm = np.linalg.norm(target_bond_vector)
        if norm > 1e-6: target_bond_vector /= norm
        new_bond_length = COVALENT_RADII.get(attach_to_atom_symbol, 0.77) + COVALENT_RADII.get(gt_anchor_symbol, 0.77)
        anchor_position = attach_to_atom_pos + target_bond_vector * new_bond_length

    rotation_matrix = _rotation_matrix_from_vectors(gt_attach_vec, target_bond_vector)
    rotated_gt_coords = gt_coords @ rotation_matrix.T
    final_coords = rotated_gt_coords - rotated_gt_coords[gt_anchor_idx] + anchor_position

    # --- START: New dihedral optimization logic for the -OH group ---
    if tuple(gt_symbols) == ('O', 'H'):
        ro_bond_axis = -target_bond_vector / np.linalg.norm(target_bond_vector)
        oxygen_pos = final_coords[0]
        hydrogen_pos_initial = final_coords[1]
        
        best_h_pos = hydrogen_pos_initial
        min_dist_initial = float('inf')
        for i, atom_pos in enumerate(atomic_coordinates):
            if i != atom_index and (mode == 'add' or i not in neighbor_indices):
                min_dist_initial = min(min_dist_initial, np.linalg.norm(hydrogen_pos_initial - atom_pos))
        best_min_dist = min_dist_initial

        for angle_deg in [60, 120, 180, 240, 300]:
            angle_rad = np.deg2rad(angle_deg)
            kmat = np.array([[0, -ro_bond_axis[2], ro_bond_axis[1]], [ro_bond_axis[2], 0, -ro_bond_axis[0]], [-ro_bond_axis[1], ro_bond_axis[0], 0]])
            rot_matrix = np.identity(3) + np.sin(angle_rad) * kmat + (1 - np.cos(angle_rad)) * kmat.dot(kmat)
            oh_vector_initial = hydrogen_pos_initial - oxygen_pos
            oh_vector_rotated = rot_matrix @ oh_vector_initial
            h_pos_candidate = oxygen_pos + oh_vector_rotated
            
            current_min_dist = float('inf')
            for i, atom_pos in enumerate(atomic_coordinates):
                if i != atom_index and (mode == 'add' or i not in neighbor_indices):
                    current_min_dist = min(current_min_dist, np.linalg.norm(h_pos_candidate - atom_pos))
            
            if current_min_dist > best_min_dist:
                best_min_dist = current_min_dist
                best_h_pos = h_pos_candidate
                
        final_coords[1] = best_h_pos
    # --- END: New dihedral optimization logic ---

    if mode == 'replace':
        atomic_symbols.pop(atom_index)
        atomic_coordinates = np.delete(atomic_coordinates, atom_index, axis=0)
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
