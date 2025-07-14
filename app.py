# app.py
import streamlit as st
from stmol import showmol
import py3Dmol
import numpy as np
from helper_functions import (
    read_xyz,
    replace_atom_with_group,
    add_group_to_atom,
    delete_atoms,
    create_xyz_string,
)

st.set_page_config(layout="wide")
st.title("3D Molecular Structure Modifier")

# --- Corrected 3D Functional Group Templates ---
groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ['C', 'H', 'H', 'H'],
            "coords": np.array([[0.000,0.000,0.000],[-0.363,1.028,0.000],[-0.363,-0.514,0.890],[-0.363,-0.514,-0.890]]),
            "anchor_index": 0, "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ['C','C','H','H','H','H','H'],
            "coords": np.array([[0.000,0.000,0.000],[1.540,0.000,0.000],[-0.363,-0.514,0.890],[-0.363,-0.514,-0.890],[1.903,0.514,0.890],[1.903,0.514,-0.890],[1.903,-1.028,0.000]]),
            "anchor_index": 0, "attachment_vector": np.array([0.363, -1.028, 0.000])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ['O', 'H'],
            "coords": np.array([
                [ 0.000,  0.000,  0.000], # O (anchor)
                [ 0.000,  0.960,  0.000], # H, positioned at a 90 degree angle
            ]),
            "anchor_index": 0, "attachment_vector": np.array([1.0, 0.0, 0.0])
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": { "symbols": ['N', 'H', 'H'], "coords": np.array([[0.000,0.000,0.000],[0.505,-0.875,0.000],[-1.010,0.000,0.000]]), "anchor_index": 0, "attachment_vector": np.array([0.505,0.875,0.000]) },
    },
    "Halogen Groups": {
        "Fluoro (-F)": {"symbols": ["F"]}, "Chloro (-Cl)": {"symbols": ["Cl"]},
        "Bromo (-Br)": {"symbols": ["Br"]}, "Iodo (-I)": {"symbols": ["I"]},
    },
}

def accept_and_continue():
    """Callback to accept the modification and update the main state."""
    if 'modified_molecule' in st.session_state and st.session_state.modified_molecule:
        st.session_state.atomic_symbols = st.session_state.modified_molecule["symbols"]
        st.session_state.atomic_coordinates = st.session_state.modified_molecule["coords"]
        st.session_state.modified_molecule = None

# Main App Logic
if 'atomic_symbols' not in st.session_state:
    st.session_state.atomic_symbols = []
    st.session_state.atomic_coordinates = np.array([])
    st.session_state.modified_molecule = None

uploaded_file = st.file_uploader("Upload Molecule (XYZ Format)", type="xyz")
if uploaded_file:
    try:
        symbols, coords = read_xyz(uploaded_file)
        st.session_state.atomic_symbols = symbols
        st.session_state.atomic_coordinates = coords
        st.session_state.modified_molecule = None # Clear old modifications
    except Exception as e:
        st.error(f"Error reading XYZ file: {e}")

if st.session_state.atomic_symbols:
    st.subheader("Current Molecule Structure")
    xyz_string = create_xyz_string(st.session_state.atomic_symbols, st.session_state.atomic_coordinates)
    
    view = py3Dmol.view(width=None, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
    view.addLabels(st.session_state.atomic_symbols, {'fontSize': 12, 'fontColor': 'black', 'backgroundColor': 'white', 'backgroundOpacity': 0.6})
    view.zoomTo()
    showmol(view, height=400)

    st.sidebar.header("Modification Controls")
    mod_type = st.sidebar.radio("Modification type:", ["Addition", "Substitution", "Deletion"], horizontal=True)
    atom_indices = list(range(len(st.session_state.atomic_symbols)))
    selected_indices = st.sidebar.multiselect("Select atom(s) to modify:", options=atom_indices, format_func=lambda i: f"Atom {i+1}: {st.session_state.atomic_symbols[i]}")
    
    group_data = None
    if mod_type in ["Substitution", "Addition"]:
        category = st.sidebar.selectbox("Group Category:", list(groups.keys()))
        group_name = st.sidebar.selectbox("Select Group:", list(groups[category].keys()))
        group_data = groups[category][group_name]
    
    if st.sidebar.button("Perform Modification", use_container_width=True, type="primary"):
        if not selected_indices:
            st.sidebar.warning("Please select at least one atom to modify.")
        else:
            try:
                symbols, coords = st.session_state.atomic_symbols.copy(), st.session_state.atomic_coordinates.copy()
                if mod_type == "Deletion":
                    symbols, coords = delete_atoms(symbols, coords, selected_indices)
                else:
                    for i in sorted(selected_indices, reverse=True):
                        if mod_type == "Substitution":
                            symbols, coords = replace_atom_with_group(symbols, coords, i, group_data)
                        elif mod_type == "Addition":
                            symbols, coords = add_group_to_atom(symbols, coords, i, group_data)
                st.session_state.modified_molecule = {"symbols": symbols, "coords": coords}
            except Exception as e:
                st.error(f"Failed to perform modification: {e}")
                st.session_state.modified_molecule = None

if 'modified_molecule' in st.session_state and st.session_state.modified_molecule:
    st.markdown("---")
    st.subheader("Modified Molecule Structure")
    mod_symbols = st.session_state.modified_molecule["symbols"]
    mod_coords = st.session_state.modified_molecule["coords"]
    xyz_mod = create_xyz_string(mod_symbols, mod_coords)
    
    view_mod = py3Dmol.view(width=None, height=400)
    view_mod.addModel(xyz_mod, "xyz")
    view_mod.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
    view_mod.addLabels(mod_symbols, {'fontSize': 12, 'fontColor': 'black', 'backgroundColor': 'white', 'backgroundOpacity': 0.6})
    view_mod.zoomTo()
    showmol(view_mod, height=400)
    
    col1, col2 = st.columns(2)
    col1.download_button("Download Modified File", xyz_mod, "modified_molecule.xyz", "text/plain", use_container_width=True)
    col2.button("Accept and Continue Editing", type="primary", use_container_width=True, on_click=accept_and_continue)
