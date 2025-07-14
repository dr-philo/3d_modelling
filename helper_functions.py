# helper_functions.py
import numpy as np
import io

COVALENT_RADII = { 'H': 0.37, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71, 'P': 1.10, 'S': 1.03, 'Cl': 0.99, 'Br': 1.14, 'I': 1.33 }

def read_xyz(file):
    content = file.getvalue().decode("utf-8").splitlines()
    if len(content) < 2: raise ValueError("Invalid XYZ file")
    num_atoms = int(content[0].strip())
    symbols, coords = [], []
    for line in content[2:2 + num_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append(list(map(float, parts[1:4])))
    return symbols, np.array(coords)

def create_xyz_string(atomic_symbols, atomic_coordinates):
    xyz_string = f"{len(atomic_symbols)}\nGenerated Molecule\n"
    for symbol, coords in zip(atomic_symbols, atomic_coordinates):
        xyz_string += f"{symbol} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n"
    return xyz_string

def _get_neighbors(atomic_symbols, atomic_coordinates, atom_index):
    neighbors = []
    for i, coords in enumerate(atomic_coordinates):
        if i == atom_index: continue
        dist = np.linalg.norm(atomic_coordinates[atom_index] - coords)
        r_sum = COVALENT_RADII.get(atomic_symbols[atom_index], 1.0) + COVALENT_RADII.get(atomic_symbols[i], 1.0)
        if dist < r_sum * 1.2:
            neighbors.append(i)
    return neighbors

def _rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)), (vec2 / np.linalg.norm(vec2))
    c, v = np.dot(a, b), np.cross(a, b)
    s = np.linalg.norm(v)
    if np.isclose(c, 1.0): return np.identity(3)
    if np.isclose(c, -1.0):
        p_axis = np.cross(a, [1,0,0])
        if np.linalg.norm(p_axis) < 1e-6: p_axis = np.cross(a, [0,1,0])
        p_axis /= np.linalg.norm(p_axis)
        kmat = np.array([[0,-p_axis[2],p_axis[1]],[p_axis[2],0,-p_axis[0]],[-p_axis[1],p_axis[0],0]])
        return np.identity(3) + 2 * kmat @ kmat
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + kmat + (kmat @ kmat) * ((1 - c) / (s**2))

def _calculate_attachment_vector(base_coords, neighbor_coords):
    if not neighbor_coords: return np.array([1.0, 0.0, 0.0])
    direction_vector = -np.sum([nc - base_coords for nc in neighbor_coords], axis=0)
    norm = np.linalg.norm(direction_vector)
    return direction_vector / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])

def _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, mode):
    if 'coords' not in group_template:
        group_template = {"symbols": group_template["symbols"], "coords": np.array([[0,0,0]]), "anchor_index": 0, "attachment_vector": [-1,0,0]}

    gt_symbols, gt_coords, gt_anchor_idx, gt_attach_vec = group_template["symbols"], group_template["coords"], group_template["anchor_index"], np.array(group_template["attachment_vector"])
    base_atom_coords = atomic_coordinates[atom_index]

    if mode == 'add':
        neighbors = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
        target_bond_vector = _calculate_attachment_vector(base_atom_coords, [atomic_coordinates[i] for i in neighbors])
        bond_length = COVALENT_RADII[atomic_symbols[atom_index]] + COVALENT_RADII[gt_symbols[gt_anchor_idx]]
        anchor_position = base_atom_coords + target_bond_vector * bond_length
    else: # replace
        neighbors = _get_neighbors(atomic_symbols, atomic_coordinates, atom_index)
        if not neighbors: raise ValueError("Cannot replace an atom with no bonds.")
        attach_to_atom_pos = atomic_coordinates[neighbors[0]]
        target_bond_vector = base_atom_coords - attach_to_atom_pos
        new_bond_length = COVALENT_RADII[atomic_symbols[neighbors[0]]] + COVALENT_RADII[gt_symbols[gt_anchor_idx]]
        anchor_position = attach_to_atom_pos + (target_bond_vector / np.linalg.norm(target_bond_vector)) * new_bond_length

    rotation = _rotation_matrix_from_vectors(gt_attach_vec, target_bond_vector)
    final_coords = (gt_coords - gt_coords[gt_anchor_idx]) @ rotation.T + anchor_position

    temp_coords = np.delete(atomic_coordinates, atom_index, axis=0) if mode == 'replace' else atomic_coordinates
    if tuple(gt_symbols) == ('O', 'H'):
        o_pos, h_initial = final_coords[0], final_coords[1]
        best_h_pos, best_dist = h_initial, min([np.linalg.norm(h_initial - p) for p in temp_coords] or [np.inf])
        for angle in np.arange(60, 360, 60):
            rot = _rotation_matrix_from_vectors(target_bond_vector, np.dot(_rotation_matrix_from_vectors(np.cross(target_bond_vector, [0,0,1]), [0,1,0]), target_bond_vector))
            axis = target_bond_vector / np.linalg.norm(target_bond_vector)
            kmat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            rot_matrix = np.identity(3) + np.sin(np.deg2rad(angle)) * kmat + (1 - np.cos(np.deg2rad(angle))) * kmat.dot(kmat)
            h_candidate = o_pos + (rot_matrix @ (h_initial - o_pos))
            dist = min([np.linalg.norm(h_candidate - p) for p in temp_coords] or [np.inf])
            if dist > best_dist:
                best_dist, best_h_pos = dist, h_candidate
        final_coords[1] = best_h_pos

    if mode == 'replace':
        final_sym = np.delete(np.array(atomic_symbols), atom_index)
        final_coord = np.delete(atomic_coordinates, atom_index, axis=0)
        final_sym = np.insert(final_sym, atom_index, gt_symbols)
        final_coord = np.insert(final_coord, atom_index, final_coords, axis=0)
        return list(final_sym.flatten()), final_coord
    else: # add
        return atomic_symbols + gt_symbols, np.vstack([atomic_coordinates, final_coords])

def add_group_to_atom(atomic_symbols, atomic_coordinates, atom_index, group_template):
    return _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, 'add')

def replace_atom_with_group(atomic_symbols, atomic_coordinates, atom_index, group_template):
    return _add_or_replace(atomic_symbols, atomic_coordinates, atom_index, group_template, 'replace')

def delete_atoms(atomic_symbols, atomic_coordinates, atom_indices):
    for index in sorted(atom_indices, reverse=True):
        del atomic_symbols[index]
        atomic_coordinates = np.delete(atomic_coordinates, index, axis=0)
    return atomic_symbols, atomic_coordinates
