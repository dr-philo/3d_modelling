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

# --- High-Quality 3D Functional Group Templates ---
# Each group has chemically accurate, pre-calculated 3D coordinates.
# anchor_index: The atom in the group that attaches to the main molecule.
# attachment_vector: A vector from the anchor atom that defines the default bonding direction.
groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ["C", "H", "H", "H"],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000],   # C (anchor)
                [-0.3630,  1.0270,  0.0000],   # H1
                [-0.3630, -0.5135,  0.8900],   # H2
                [-0.3630, -0.5135, -0.8900],   # H3
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.0, 0.0, 0.0]) # Points from group towards molecule
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ["C", "C", "H", "H", "H", "H", "H"],
            "coords": np.array([
                # This is a staggered conformation of ethyl
                [ 0.0000,  0.0000,  0.0000], # C1 (anchor, -CH2-)
                [ 1.5200,  0.0000,  0.0000], # C2 (-CH3)
                [-0.4500,  1.0000,  0.0000], # H on C1
                [-0.4500, -0.5000,  0.8660], # H on C1
                [ 1.9700,  0.0000,  1.0000], # H on C2
                [ 1.9700,  0.8660, -0.5000], # H on C2
                [ 1.9700, -0.8660, -0.5000], # H on C2
            ]),
            "anchor_index": 0,
            # Vector pointing from C1 towards where the rest of the molecule (R) would be
            "attachment_vector": np.array([-0.75, -0.2, -0.3])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ["O", "H"],
            "coords": np.array([
                [0.000,  0.000, 0.000],      # O (anchor)
                [0.318,  0.905, 0.000],      # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ["N", "H", "H"],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # N (anchor)
                [-0.3300,  0.9500,  0.0000], # H1
                [-0.3300, -0.4750,  0.8227], # H2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([1.0, 0.0, 0.0])
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
        st.session_state['atomic_symbols'] = atomic_symbols
        st.session_state['atomic_coordinates'] = atomic_coordinates
    except ValueError as e:
        st.error(f"Error reading XYZ file: {e}")
        st.stop()

if 'atomic_symbols' in st.session_state:
    atomic_symbols = st.session_state['atomic_symbols']
    atomic_coordinates = st.session_state['atomic_coordinates']

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
    
    mod_type = st.sidebar.radio("Modification type:", ["Addition", "Substitution", "Deletion"])

    atom_positions = list(range(1, len(atomic_symbols) + 1))
    selected_positions = st.sidebar.multiselect(
        f"Select atom(s) to modify:",
        options=atom_positions,
        format_func=lambda x: f"Atom {x}: {atomic_symbols[x-1]}",
    )
    
    group_data = None
    if mod_type in ["Substitution", "Addition"]:
        group_category = st.sidebar.selectbox("Select functional group category:", list(groups.keys()))
        selected_group_name = st.sidebar.selectbox("Select group:", list(groups[group_category].keys()))
        group_data = groups[group_category][selected_group_name]
    
    if st.sidebar.button("Perform Modification", use_container_width=True):
        if not selected_positions:
            st.sidebar.warning("Please select at least one atom to modify.")
        else:
            new_atomic_symbols = atomic_symbols.copy()
            new_atomic_coordinates = atomic_coordinates.copy()

            if mod_type == "Deletion":
                new_atomic_symbols, new_atomic_coordinates = delete_atoms(
                    new_atomic_symbols, new_atomic_coordinates, [p - 1 for p in selected_positions]
                )
            else:
                # Process modifications in reverse index order to avoid shifting issues
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
            
            # Update the session state to reflect the modified structure
            st.session_state['atomic_symbols'] = new_atomic_symbols
            st.session_state['atomic_coordinates'] = new_atomic_coordinates
            st.rerun()
    view_mod.zoomTo()
    showmol(view_mod, height=400, width=800)
    # Create a download button for the displayed structure
    st.download_button(
        label="Download Current XYZ File",
        data=create_xyz_string(atomic_symbols, atomic_coordinates),
        file_name="current_molecule.xyz",
        mime="text/plain",
    )
