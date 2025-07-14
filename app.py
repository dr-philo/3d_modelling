# app.py
import streamlit as st
from stmol import showmol
import py3Dmol
import numpy as np
from helper_functions import (
    read_xyz,
    # write_xyz is not directly used in app.py, can be removed from import
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
            "coords": np.array([
                [ 0.000,  0.000,  0.000], # C (anchor)
                [-0.363,  1.028,  0.000], # H1
                [-0.363, -0.514,  0.890], # H2
                [-0.363, -0.514, -0.890], # H3
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ['C', 'C', 'H', 'H', 'H', 'H', 'H'],
            "coords": np.array([
                [ 0.000,  0.000,  0.000], # C1 (anchor)
                [ 1.540,  0.000,  0.000], # C2
                [-0.363, -0.514,  0.890], # H on C1
                [-0.363, -0.514, -0.890], # H on C1
                [ 1.903,  0.514,  0.890], # H on C2
                [ 1.903,  0.514, -0.890], # H on C2
                [ 1.903, -1.028,  0.000], # H on C2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.363, -1.028, 0.000])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ['O', 'H'],
            "coords": np.array([
                [ 0.000,  0.000,  0.000], # O (anchor)
                [-0.240,  0.930,  0.000], # H, positioned to create a 104.5 degree angle
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.0, 0.0, 0.0]) # Inward vector for the R-O bond
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ['N', 'H', 'H'],
            "coords": np.array([
                [0.000,  0.000,  0.000], # N (anchor)
                [1.010,  0.000,  0.000], # H1
                [-0.337, 0.952,  0.000], # H2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.337, 0.476, -0.824])
        },
    },
    "Halogen Groups": {
        "Fluoro (-F)": {"symbols": ["F"]}, "Chloro (-Cl)": {"symbols": ["Cl"]},
        "Bromo (-Br)": {"symbols": ["Br"]}, "Iodo (-I)": {"symbols": ["I"]},
    },
}

# --- Streamlit UI and State Management ---

def accept_and_continue():
    """Callback function to accept the modification and update the main state."""
    mod_data = st.session_state.modified_molecule
    if mod_data:
        st.session_state.atomic_symbols = mod_data["symbols"]
        st.session_state.atomic_coordinates = mod_data["coords"]
        st.session_state.modified_molecule = None

# --- Main App Logic ---

uploaded_file = st.file_uploader("Upload Molecule (XYZ Format)", type="xyz")

if uploaded_file is not None:
    try:
        atomic_symbols, atomic_coordinates = read_xyz(uploaded_file)
        st.session_state.atomic_symbols = atomic_symbols
        st.session_state.atomic_coordinates = atomic_coordinates
        st.session_state.modified_molecule = None
    except ValueError as e:
        st.error(f"Error reading XYZ file: {e}")
        st.stop()

if 'atomic_symbols' in st.session_state:
    atomic_symbols = st.session_state.atomic_symbols
    atomic_coordinates = st.session_state.atomic_coordinates

    st.subheader("Current Molecule Structure")
    xyz_string = create_xyz_string(atomic_symbols, atomic_coordinates)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        view.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                                "fontSize": 12, "fontColor": "black", "backgroundColor": "white",
                                "backgroundOpacity": 0.6})
    view.zoomTo()
    showmol(view, height=400, width=800)

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
                st.session_state.modified_molecule = {
                    "symbols": new_atomic_symbols, "coords": new_atomic_coordinates
                }
            except Exception as e:
                st.error(f"Failed to perform modification: {e}")
                st.session_state.modified_molecule = None

if 'modified_molecule' in st.session_state and st.session_state.modified_molecule:
    st.markdown("---")
    st.subheader("Modified Molecule Structure")

    mod_symbols = st.session_state.modified_molecule["symbols"]
    mod_coords = st.session_state.modified_molecule["coords"]
    
    xyz_string_mod = create_xyz_string(mod_symbols, mod_coords)

    view_mod = py3Dmol.view(width=800, height=400)# app.py
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

groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ['C', 'H', 'H', 'H'],
            "coords": np.array([[0.000,0.000,0.000],[-0.363,1.028,0.000],[-0.363,-0.514,0.890],[-0.363,-0.514,-0.890]]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ['C','C','H','H','H','H','H'],
            "coords": np.array([[0.000,0.000,0.000],[1.540,0.000,0.000],[-0.363,-0.514,0.890],[-0.363,-0.514,-0.890],[1.903,0.514,0.890],[1.903,0.514,-0.890],[1.903,-1.028,0.000]]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.363, -1.028, 0.000])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ['O', 'H'],
            "coords": np.array([[0.000,0.000,0.000],[-0.320,0.905,0.000]]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.0, 0.0, 0.0])
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ['N', 'H', 'H'],
            "coords": np.array([[0.000,0.000,0.000],[1.010,0.000,0.000],[-0.337,0.952,0.000]]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.337, 0.476, -0.824])
        },
    },
    "Halogen Groups": {
        "Fluoro (-F)": {"symbols": ["F"]}, "Chloro (-Cl)": {"symbols": ["Cl"]},
        "Bromo (-Br)": {"symbols": ["Br"]}, "Iodo (-I)": {"symbols": ["I"]},
    },
}

def accept_and_continue():
    mod_data = st.session_state.modified_molecule
    if mod_data:
        st.session_state.atomic_symbols = mod_data["symbols"]
        st.session_state.atomic_coordinates = mod_data["coords"]
        st.session_state.modified_molecule = None

uploaded_file = st.file_uploader("Upload Molecule (XYZ Format)", type="xyz")

if uploaded_file is not None:
    try:
        st.session_state.atomic_symbols, st.session_state.atomic_coordinates = read_xyz(uploaded_file)
        st.session_state.modified_molecule = None
    except Exception as e:
        st.error(f"Error reading XYZ file: {e}")
        st.stop()

if 'atomic_symbols' in st.session_state:
    st.subheader("Current Molecule Structure")
    xyz_string = create_xyz_string(st.session_state.atomic_symbols, st.session_state.atomic_coordinates)
    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
    for i, (symbol, coords) in enumerate(zip(st.session_state.atomic_symbols, st.session_state.atomic_coordinates)):
        view.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]}, "fontSize": 12, "fontColor": "black", "backgroundColor": "white", "backgroundOpacity": 0.6})
    view.zoomTo()
    showmol(view, height=400, width=800)

    st.sidebar.header("Modification Controls")
    mod_type = st.sidebar.radio("Modification type:", ["Addition", "Substitution", "Deletion"], horizontal=True)
    atom_positions = list(range(1, len(st.session_state.atomic_symbols) + 1))
    selected_positions = st.sidebar.multiselect("Select atom(s) to modify:", options=atom_positions, format_func=lambda x: f"Atom {x}: {st.session_state.atomic_symbols[x-1]}")
    
    group_data = None
    if mod_type in ["Substitution", "Addition"]:
        group_category = st.sidebar.selectbox("Functional Group Category:", list(groups.keys()))
        selected_group_name = st.sidebar.selectbox("Select Group:", list(groups[group_category].keys()))
        group_data = groups[group_category][selected_group_name]
    
    if st.sidebar.button("Perform Modification", use_container_width=True, type="primary"):
        if not selected_positions:
            st.sidebar.warning("Please select at least one atom to modify.")
        else:
            try:
                symbols, coords = st.session_state.atomic_symbols, st.session_state.atomic_coordinates
                if mod_type == "Deletion":
                    symbols, coords = delete_atoms(symbols, coords, [p - 1 for p in selected_positions])
                else:
                    for pos in sorted(selected_positions, reverse=True):
                        if mod_type == "Substitution":
                            symbols, coords = replace_atom_with_group(symbols, coords, pos - 1, group_data)
                        elif mod_type == "Addition":
                            symbols, coords = add_group_to_atom(symbols, coords, pos - 1, group_data)
                st.session_state.modified_molecule = {"symbols": symbols, "coords": coords}
            except Exception as e:
                st.error(f"Failed to perform modification: {e}")
                st.session_state.modified_molecule = None

    if 'modified_molecule' in st.session_state and st.session_state.modified_molecule:
        st.markdown("---")
        st.subheader("Modified Molecule Structure")
        mod_symbols = st.session_state.modified_molecule["symbols"]
        mod_coords = st.session_state.modified_molecule["coords"]
        xyz_string_mod = create_xyz_string(mod_symbols, mod_coords)
        
        view_mod = py3Dmol.view(width=800, height=400)
        view_mod.addModel(xyz_string_mod, "xyz")
        view_mod.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
        for i, (symbol, coords) in enumerate(zip(mod_symbols, mod_coords)):
            view_mod.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]}, "fontSize": 12, "fontColor": "black", "backgroundColor": "white", "backgroundOpacity": 0.6})
        view_mod.zoomTo()
        showmol(view_mod, height=400, width=800)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("Download Modified File", xyz_string_mod, "modified_molecule.xyz", "text/plain", use_container_width=True)
        with col2:
            st.button("Accept and Continue Editing", type="primary", use_container_width=True, on_click=accept_and_continue)
    view_mod.addModel(xyz_string_mod, "xyz")
    view_mod.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
    for i, (symbol, coords) in enumerate(zip(mod_symbols, mod_coords)):
        view_mod.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                                "fontSize": 12, "fontColor": "black", "backgroundColor": "white",
                                "backgroundOpacity": 0.6})
    view_mod.zoomTo()
    showmol(view_mod, height=400, width=800)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Modified File", data=xyz_string_mod,
            file_name="modified_molecule.xyz", mime="text/plain", use_container_width=True
        )
    with col2:
        st.button("Accept and Continue Editing", type="primary", use_container_width=True,
                  on_click=accept_and_continue)
