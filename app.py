# app.py
import streamlit as st
from stmol import showmol
import py3Dmol
import numpy as np
from helper_functions import (
    read_xyz,
    write_xyz,
    replace_atom_with_group,
    add_group_to_atom,
    delete_atoms,
    create_xyz_string,
)

st.set_page_config(layout="wide")
st.title("3D Molecular Structure Modifier")

# --- High-Quality 3D Functional Group Templates (Generated with RDKit) ---
groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ['C', 'H', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # C (anchor)
                [ 0.6310,  0.8924,  0.0000], # H
                [-0.6310,  0.8924,  0.0000], # H
                [ 0.0000, -0.0000,  1.0930]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.0000, -1.5000, -0.0000]) # Points towards molecule
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ['C', 'C', 'H', 'H', 'H', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # C1 (anchor, -CH2-)
                [-1.5270,  0.0000,  0.0000], # C2 (-CH3)
                [ 0.4431, -0.5361, -0.8718], # H on C1
                [ 0.4431,  1.0543, -0.1191], # H on C1
                [-1.9642, -0.5130,  0.8800], # H on C2
                [-1.9642,  1.0425,  0.1082], # H on C2
                [-1.6338, -0.5295, -0.8882]  # H on C2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.2821, -0.1633, 0.7710]) # Points towards molecule
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ['O', 'H'],
            "coords": np.array([
                [ 0.0000, 0.0000, 0.0000], # O (anchor)
                [-0.3752, 0.8871, 0.0000]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.3621, -0.1551, 0.0000])
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ['N', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # N (anchor)
                [ 0.5988,  0.8163,  0.0000], # H
                [-0.5988,  0.8163,  0.0000]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-0.0000, -1.4582, 0.0000])
        },
    },
    "Halogen Groups": {
        "Fluoro (-F)": {"symbols": ["F"]}, "Chloro (-Cl)": {"symbols": ["Cl"]},
        "Bromo (-Br)": {"symbols": ["Br"]}, "Iodo (-I)": {"symbols": ["I"]},
    },
}

# --- Streamlit UI ---

uploaded_file = st.file_uploader("Upload Molecule (XYZ Format)", type="xyz")

if uploaded_file is not None:
    try:
        atomic_symbols, atomic_coordinates = read_xyz(uploaded_file)
        # Initialize session state for both current and modified molecules
        st.session_state['atomic_symbols'] = atomic_symbols
        st.session_state['atomic_coordinates'] = atomic_coordinates
        st.session_state.modified_molecule = None # Clear any previous modifications
    except ValueError as e:
        st.error(f"Error reading XYZ file: {e}")
        st.stop()

if 'atomic_symbols' in st.session_state:
    atomic_symbols = st.session_state['atomic_symbols']
    atomic_coordinates = st.session_state['atomic_coordinates']

    st.subheader("Current Molecule Structure")
    xyz_string = create_xyz_string(atomic_symbols, atomic_coordinates)

    # Display 3D structure
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        view.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                                "fontSize": 12, "fontColor": "black", "backgroundColor": "white",
                                "backgroundOpacity": 0.6})
    view.zoomTo()
    showmol(view, height=400, width=800)

    # --- Sidebar Controls ---
    st.sidebar.header("Modification Controls")
    mod_type = st.sidebar.radio("Modification type:", ["Addition", "Substitution", "Deletion"], horizontal=True)
    atom_positions = list(range(1, len(atomic_symbols) + 1))
    selected_positions = st.sidebar.multiselect(
        f"Select atom(s) to modify:", options=atom_positions,
        format_func=lambda x: f"Atom {x}: {atomic_symbols[x-1]}"
    )
    group_data = None
    if mod_type in ["Substitution", "Addition"]:
        group_category = st.sidebar.selectbox("Functional Group Category:", list(groups.keys()))
        selected_group_name = st.sidebar.selectbox("Select Group:", list(groups[group_category].keys()))
        group_data = groups[group_category][selected_group_name]
    
    if st.sidebar.button("Perform Modification", use_container_width=True, type="primary"):
        if not selected_positions:
            st.sidebar.warning("Please select at least one atom to modify.")
        else:
            new_atomic_symbols = atomic_symbols.copy()
            new_atomic_coordinates = atomic_coordinates.copy()
            try:
                if mod_type == "Deletion":
                    new_atomic_symbols, new_atomic_coordinates = delete_atoms(
                        new_atomic_symbols, new_atomic_coordinates, [p - 1 for p in selected_positions]
                    )
                else:
                    for pos in sorted(selected_positions, reverse=True):
                        atom_index = pos - 1
                        if mod_type == "Substitution":
                            new_atomic_symbols, new_atomic_coordinates = replace_atom_with_group(
                                new_atomic_symbols, new_atomic_coordinates, atom_index, group_data
                            )
                        elif mod_type == "Addition":
                            new_atomic_symbols, new_atomic_coordinates = add_group_to_atom(
                                new_atomic_symbols, new_atomic_coordinates, atom_index, group_data
                            )
                # Store the successful modification in session state to be displayed
                st.session_state.modified_molecule = {
                    "symbols": new_atomic_symbols, "coords": new_atomic_coordinates
                }
            except Exception as e:
                st.error(f"Failed to perform modification: {e}")
                st.session_state.modified_molecule = None

# --- Display Modified Structure Section (if it exists) ---
if 'modified_molecule' in st.session_state and st.session_state.modified_molecule:
    st.markdown("---")
    st.subheader("Modified Molecule Structure")

    mod_symbols = st.session_state.modified_molecule["symbols"]
    mod_coords = st.session_state.modified_molecule["coords"]
    
    xyz_string_mod = create_xyz_string(mod_symbols, mod_coords)

    # Display 3D structure
    view_mod = py3Dmol.view(width=800, height=400)
    view_mod.addModel(xyz_string_mod, "xyz")
    view_mod.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
    for i, (symbol, coords) in enumerate(zip(mod_symbols, mod_coords)):
        view_mod.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                                "fontSize": 12, "fontColor": "black", "backgroundColor": "white",
                                "backgroundOpacity": 0.6})
    view_mod.zoomTo()
    showmol(view_mod, height=400, width=800)

    # Create two columns for the action buttons below the modified structure
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Modified File", data=xyz_string_mod,
            file_name="modified_molecule.xyz", mime="text/plain", use_container_width=True
        )
    with col2:
        if st.button("Accept and Continue Editing", type="primary", use_container_width=True):
            # Promote the modified structure to the main "current" state
            st.session_state.atomic_symbols = mod_symbols
            st.session_state.atomic_coordinates = mod_coords
            # Clear the temporary modified state
            st.session_state.modified_molecule = None
            st.rerun()
