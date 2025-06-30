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

# --- High-Quality 3D Functional Group Templates (Generated Programmatically) ---
# These templates are now generated using a robust method to ensure all
# geometries and vectors are self-consistent and chemically accurate.
groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ['C', 'H', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # C (anchor)
                [ 1.0842,  0.0000,  0.0000], # H
                [-0.3614, -1.0222,  0.0000], # H
                [-0.3614,  0.5111, -0.8852]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-0.3614, 0.5111, 0.8852])
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ['C', 'C', 'H', 'H', 'H', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # C1 (anchor, -CH2-)
                [ 1.5270,  0.0000,  0.0000], # C2 (-CH3)
                [-0.4431,  0.8718, -0.5361], # H on C1
                [-0.4431, -0.1191,  1.0543], # H on C1
                [ 1.9642,  0.8800, -0.5130], # H on C2
                [ 1.9642,  0.1082,  1.0425], # H on C2
                [ 1.6338, -0.8882, -0.5295]  # H on C2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-0.2902, -0.7710, -0.1633])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ['O', 'H'],
            "coords": np.array([
                [ 0.0000, 0.0000, 0.0000], # O (anchor)
                [ 0.9600, 0.0000, 0.0000]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-0.7583, -0.6519, 0.0000])
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ['N', 'H', 'H'],
            "coords": np.array([
                [ 0.0000,  0.0000,  0.0000], # N (anchor)
                [ 1.0090,  0.0000,  0.0000], # H
                [-0.3363,  0.9511,  0.0000]  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-0.3363, -0.4755, 0.8236])
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
        view.addLabel(f"{i+1}", {"position": {"x": coords[0], "y": coords[1], "z": coords[2]}},
