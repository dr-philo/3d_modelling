# helper_functions.py
import numpy as np
import io

# Covalent radii in Angstroms for common elements
COVALENT_RADII = {
    'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71,
    'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33,
}

def read_xyz(file):
    """Reads molecular structure from XYZ file format."""
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

def create_xyz_string(atomic_symbols, atomic_coordinates):
    """Creates an XYZ format string for visualization."""
    xyz_string = f"{len(atomic_symbols)}\nModified molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string

def _get_neighbors(atomic_symbols, atomic_coordinates, atom_index):
    """Identifies the bonded neighbors of a given atom based on covalent radii."""
    neighbors = []
    base_coords = atomic_coordinates[atom_index]
    base_symbol = atomic_symbols[atom_index]
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        if i == atom_index: continue
        expected_bond_length = COVALENT_RADII.get(base_symbol, 1.0) + COVALENT_RADII.get(symbol, 1.0)
        if np.linalg.norm(base_coords - coords) < expected_bond_length * 1.2:
            neighbors.append(i)
    return neighbors

def _rotation_matrix_from_vectors(vec1, vec2):
    """Finds the rotation matrix that aligns vec1 to vec2."""
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    c = np.dot(a, b)

    if np.isclose(c, 1.0): return np.identity(3)

    if np.isclose(c, -1.0):
        p_axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(p_axis) < 1e-6: p_axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        p_axis /= np.linalg.norm(p_axis)
        kmat = np.array([[0, -p_axis[2], p_axis[1]], [p_axis[2], 0, -p_axis[0]], [-p_axis[1], p_axis[0], 0]])
        return np.identity(3) + 2 * kmat.dot(kmat)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def _calculate_attachment_vector(base_atom_coords, neighbor_coords_list):
    """Calculates the ideal direction vector for attaching a new group."""
    if not neighbor_coords_list: return np.array([1.0, 0.0, 0.0])
    sum_of_vectors = np.sum([np.array(nc) - base_atom_coords for nc in neighbor_coords_list], axis=0)
    direction_vector = -sum_of_vectors
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

def _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, mode):
    """Core function to handle both additions and substitutions using 3D templates."""
    if 'coords' not in group_template:
        group_template = {"symbols": group_template["symbols"], "coords": np.array([[0.0, 0.0, 0.0]]), "anchor_index": 0, "attachment_vector": np.array([-1.0, 0.0, 0.0])}

    base_atom_coords = atomic_coordinates[atom_index]
    gt_symbols, gt_coords, gt_anchor_idx, gt_attach_vec = group_template["symbols"], group_template["coords"], group_template["anchor_index"], group_template["attachment_vector"]
    gt_anchor_symbol = gt_symbols[gt_anchor_idx]
    
    neighbor_indices = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)

    if mode == 'add':
        target_bond_vector = _calculate_attachment_vector(base_atom_coords, [atomic_coordinates[i] for i in neighbor_indices])
        bond_length = COVALENT_RADII.get(atomic_symbols[atom_index], 0.77) + COVALENT_RADII.get(gt_anchor_symbol, 0.77)
        anchor_position = base_atom_coords + target_bond_vector * bond_length
    else: # 'replace'
        if not neighbor_indices: raise ValueError("Cannot replace an atom with no bonded neighbors to attach to.")
        attach_to_atom_pos = atomic_coordinates[neighbor_indices[0]]
        attach_to_atom_symbol = atomic_symbols[neighbor_indices[0]]
        target_bond_vector = base_atom_coords - attach_to_atom_pos
        norm = np.linalg.norm(target_bond_vector)
        if norm > 1e-6: target_bond_vector /= norm
        new_bond_length = COVALENT_RADII.get(attach_to_atom_symbol, 0.77) + COVALENT_RADII.get(gt_anchor_symbol, 0.77)
        anchor_position = attach_to_atom_pos + target_bond_vector * new_bond_length

    rotation_matrix = _rotation_matrix_from_vectors(gt_attach_vec, target_bond_vector)
    rotated_gt_coords = gt_coords @ rotation_matrix.T
    final_coords = rotated_gt_coords - rotated_gt_coords[gt_anchor_idx] + anchor_position

    # --- START: Dihedral optimization logic for the -OH group ---
    if tuple(gt_symbols) == ('O', 'H'):
        temp_coords = atomic_coordinates.copy()
        if mode == 'replace': temp_coords = np.delete(temp_coords, atom_index, axis=0)
        
        ro_bond_axis = -target_bond_vector / np.linalg.norm(target_bond_vector)
        oxygen_pos, hydrogen_pos_initial = final_coords[0], final_coords[1]
        
        best_h_pos = hydrogen_pos_initial
        min_dist_initial = min([np.linalg.norm(hydrogen_pos_initial - pos) for pos in temp_coords] or [float('inf')])
        best_min_dist = min_dist_initial

        for angle_deg in np.arange(60, 360, 60):
            angle_rad = np.deg2rad(angle_deg)
            axis = ro_bond_axis
            kmat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rot_matrix = np.identity(3) + np.sin(angle_rad) * kmat + (1 - np.cos(angle_rad)) * kmat.dot(kmat)
            oh_vector_rotated = rot_matrix @ (hydrogen_pos_initial - oxygen_pos)
            h_pos_candidate = oxygen_pos + oh_vector_rotated
            
            current_min_dist = min([np.linalg.norm(h_pos_candidate - pos) for pos in temp_coords] or [float('inf')])
            
            if current_min_dist > best_min_dist:
                best_min_dist, best_h_pos = current_min_dist, h_pos_candidate
                
        final_coords[1] = best_h_pos
    # --- END: Dihedral optimization logic ---

    if mode == 'replace':
        atomic_symbols.pop(atom_index)
        atomic_coordinates = np.delete(atomic_coordinates, atom_index, axis=0)
    
    new_symbols = atomic_symbols + gt_symbols
    new_coords = np.vstack([atomic_coordinates, final_coords]) if len(atomic_coordinates) > 0 else final_coords
    
    return new_symbols, new_coords

def add_group_to_atom(atomic_symbols, atomic_coordinates, atom_index, group_template):
    return _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, 'add')

def replace_atom_with_group(atomic_symbols, atomic_coordinates, atom_index, group_template):
    # For replacement, we split the logic to handle list/array manipulation correctly
    symbols_copy, coords_copy = atomic_symbols.copy(), atomic_coordinates.copy()
    temp_symbols, temp_coords = _add_or_replace(symbols_copy, coords_copy, atom_index, group_template, 'replace')
    
    # Re-insert the new group at the correct position
    original_size = len(atomic_symbols)
    group_size = len(group_template["symbols"])
    
    final_symbols = temp_symbols[:atom_index] + temp_symbols[original_size-1:]
    final_coords = np.vstack([temp_coords[:atom_index], temp_coords[original_size-1:]])
    
    return final_symbols, final_coords

def delete_atoms(atomic_symbols, atomic_coordinates, atom_indices):
    """Deletes atoms at the specified indices."""
    for index in sorted(atom_indices, reverse=True):
        if 0 <= index < len(atomic_symbols):
            del atomic_symbols[index]
            atomic_coordinates = np.delete(atomic_coordinates, index, axis=0)
    return atomic_symbols, atomic_coordinates
