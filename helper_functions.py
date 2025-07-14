# helper_functions.py
import numpy as np
import io

COVALENT_RADII = { 'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, 'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33 }

def read_xyz(file):
    """Reads molecular structure from XYZ file format."""
    content = file.getvalue().decode("utf-8").splitlines()
    if len(content) < 2: raise ValueError("Invalid XYZ file")
    try:
        num_atoms = int(content[0].strip())
    except (ValueError, IndexError):
        raise ValueError("Invalid XYZ format: Cannot read atom count.")
    
    symbols, coords = [], []
    for line in content[2:2 + num_atoms]:
        parts = line.split()
        if len(parts) < 4: raise ValueError("Invalid XYZ format: Malformed atom line.")
        symbols.append(parts[0])
        coords.append([float(c) for c in parts[1:4]])
    return symbols, np.array(coords)

def create_xyz_string(atomic_symbols, atomic_coordinates):
    """Creates an XYZ format string for visualization."""
    xyz_string = f"{len(atomic_symbols)}\nGenerated Molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string

def _get_neighbors(symbols, coords, atom_index):
    """Identifies the bonded neighbors of a given atom."""
    neighbors = []
    for i, other_coords in enumerate(coords):
        if i == atom_index: continue
        dist = np.linalg.norm(coords[atom_index] - other_coords)
        r_sum = COVALENT_RADII.get(symbols[atom_index], 1.0) + COVALENT_RADII.get(symbols[i], 1.0)
        if dist < r_sum * 1.2:
            neighbors.append(i)
    return neighbors

def _rotation_matrix_from_vectors(vec1, vec2):
    """Finds the rotation matrix that aligns vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    c, v = np.dot(a, b), np.cross(a, b)
    s = np.linalg.norm(v)
    if np.isclose(c, 1.0): return np.identity(3)
    if np.isclose(c, -1.0):
        p_axis = np.cross(a, [1,0,0]) if np.linalg.norm(np.cross(a, [1,0,0])) > 1e-6 else np.cross(a, [0,1,0])
        p_axis /= np.linalg.norm(p_axis)
        kmat = np.array([[0,-p_axis[2],p_axis[1]],[p_axis[2],0,-p_axis[0]],[-p_axis[1],p_axis[0],0]])
        return np.identity(3) + 2 * kmat @ kmat
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + kmat + (kmat @ kmat) * ((1 - c) / (s**2))

def _calculate_attachment_vector(base_coords, neighbor_coords):
    """Calculates the ideal direction vector for attaching a new group."""
    if not neighbor_coords: return np.array([1.0, 0.0, 0.0])
    direction = -sum(np.array(nc) - base_coords for nc in neighbor_coords)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

def _add_or_replace(symbols, coords, atom_index, group_template, mode):
    """Core logic to add a group or replace an atom with a group."""
    if 'coords' not in group_template:
        group_template.update({"coords": np.array([[0,0,0]]), "anchor_index": 0, "attachment_vector": [-1,0,0]})

    gt_sym, gt_coords, gt_anchor, gt_vec = (group_template["symbols"], group_template["coords"],
                                            group_template["anchor_index"], np.array(group_template["attachment_vector"]))
    base_coords = coords[atom_index]

    if mode == 'add':
        neighbors = _get_neighbors(symbols, coords, atom_index)
        target_vec = _calculate_attachment_vector(base_coords, [coords[i] for i in neighbors])
        bond_len = COVALENT_RADII[symbols[atom_index]] + COVALENT_RADII[gt_sym[gt_anchor]]
        anchor_pos = base_coords + target_vec * bond_len
    else:  # replace
        neighbors = _get_neighbors(symbols, coords, atom_index)
        if not neighbors: raise ValueError("Cannot replace an atom with no bonds.")
        attach_to_pos = coords[neighbors[0]]
        target_vec = base_coords - attach_to_pos
        bond_len = COVALENT_RADII[symbols[neighbors[0]]] + COVALENT_RADII[gt_sym[gt_anchor]]
        anchor_pos = attach_to_pos + (target_vec / np.linalg.norm(target_vec)) * bond_len

    rotation = _rotation_matrix_from_vectors(gt_vec, target_vec)
    final_coords = (gt_coords - gt_coords[gt_anchor]) @ rotation.T + anchor_pos
    
    if mode == 'replace':
        symbols.pop(atom_index)
        coords = np.delete(coords, atom_index, axis=0)
        for i, sym in enumerate(gt_sym):
            symbols.insert(atom_index + i, sym)
            coords = np.insert(coords, atom_index + i, final_coords[i], axis=0)
    else:  # add
        symbols.extend(gt_sym)
        coords = np.vstack([coords, final_coords])
        
    return symbols, coords

def add_group_to_atom(s, c, i, g): return _add_or_replace(s, c, i, g, 'add')
def replace_atom_with_group(s, c, i, g): return _add_or_replace(s, c, i, g, 'replace')
def delete_atoms(s, c, indices):
    for i in sorted(indices, reverse=True):
        del s[i]
        c = np.delete(c, i, axis=0)
    return s, c
