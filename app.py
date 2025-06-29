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
st.title("Structural Modification of Molecules with 3D Group Templates")

# --- 3D Functional Group Templates ---
# Each group has:
# - symbols: List of atomic symbols.
# - coords: An array of 3D coordinates for the group's internal geometry.
# - anchor_index: The index of the atom in 'symbols' that attaches to the main molecule.
# - attachment_vector: A vector from the anchor atom that defines the bonding direction.
groups = {
    "Alkyl Groups": {
        "Methyl (-CH3)": {
            "symbols": ["C", "H", "H", "H"],
            "coords": np.array([
                [0.000, 0.000, 0.000],  # C (anchor)
                [0.629, 0.890, 0.000],  # H
                [-1.020, 0.400, 0.000], # H
                [0.391, -1.290, 0.000]  # H (These are 2D for simplicity, will be rotated into 3D)
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([0.0, 0.0, -1.0]) # A placeholder vector to be oriented
        },
        "Ethyl (-CH2CH3)": {
            "symbols": ["C", "C", "H", "H", "H", "H", "H"],
            "coords": np.array([
                [0.000, 0.000, 0.000],   # C1 (anchor)
                [1.540, 0.000, 0.000],   # C2
                [-0.450, 1.000, 0.000],  # H on C1
                [-0.450, -1.000, 0.000], # H on C1
                [1.990, 0.500, -0.866],  # H on C2
                [1.990, -1.000, 0.000],  # H on C2
                [1.990, 0.500, 0.866]    # H on C2
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
    },
    "Oxygen-containing Groups": {
        "Hydroxyl (-OH)": {
            "symbols": ["O", "H"],
            "coords": np.array([
                [0.000, 0.000, 0.000],  # O (anchor)
                [0.960, 0.000, 0.000],  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0]) # R-O bond direction
        },
    },
    "Nitrogen-containing Groups": {
        "Amino (-NH2)": {
            "symbols": ["N", "H", "H"],
            "coords": np.array([
                [0.000, 0.000, 0.000],   # N (anchor)
                [0.587, 0.824, 0.000],   # H
                [0.587, -0.824, 0.000],  # H
            ]),
            "anchor_index": 0,
            "attachment_vector": np.array([-1.0, 0.0, 0.0])
        },
    },
    "Halogen Groups": {
        "Fluoro (-F)": {"symbols": ["F"]}, "Chloro (-Cl)": {"symbols": ["Cl"]},
        "Bromo (-Br)": {"symbols": ["Br"]}, "Iodo (-I)": {"symbols": ["I"]},
    },
}


# File upload in the main area
uploaded_file = st.file_uploader("Upload XYZ File", type="xyz")

if uploaded_file is not None:
    try:
        atomic_symbols, atomic_coordinates = read_xyz(uploaded_file)
        st.session_state['atomic_symbols'] = atomic_symbols
        st.session_state['atomic_coordinates'] = atomic_coordinates
    except ValueError as e:
        st.error(e)
        st.stop()


if 'atomic_symbols' in st.session_state:
    # ... (The rest of the app.py code is identical to the previous version)
    # This includes the UI for displaying the molecule and handling modifications.
    # No changes are needed here as the logic is now handled by the new helper functions.
    atomic_symbols = st.session_state['atomic_symbols']
    atomic_coordinates = st.session_state['atomic_coordinates']

    # Display original structure with annotations
    st.subheader("Original Molecule Structure")
    xyz_string = create_xyz_string(atomic_symbols, atomic_coordinates)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({
        'sphere': {'radius': 0.3},
        'stick': {'radius': 0.15}
    })
    # Add labels to atoms
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        view.addLabel(
            f"{i+1}",
            {
                "position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                "fontSize": 12,
                "fontColor": "black",
                "backgroundColor": "white",
                "backgroundOpacity": 0.5,
            },
        )

    view.zoomTo()
    showmol(view, height=400, width=800)

    # Sidebar for modification controls
    st.sidebar.header("Modification Controls")
    
    # Using session state to manage modifications
    if 'modifications' not in st.session_state:
        st.session_state.modifications = []

    def add_modification():
        st.session_state.modifications.append({'type': 'Addition', 'atoms': [], 'group_cat': list(groups.keys())[0], 'group_sel': list(groups[list(groups.keys())[0]].keys())[0]})

    st.sidebar.button("Add another modification", on_click=add_modification)

    mods_to_apply = []
    indices_to_delete = []

    for i, mod in enumerate(st.session_state.modifications):
        st.sidebar.subheader(f"Modification {i + 1}")
        
        mod_type = st.sidebar.radio(
            "Modification type:",
            ["Addition", "Substitution", "Deletion"],
            index=["Addition", "Substitution", "Deletion"].index(mod.get('type', 'Addition')),
            key=f"mod_type_{i}",
        )

        atom_positions = list(range(1, len(atomic_symbols) + 1))
        
        selected_positions = st.sidebar.multiselect(
            f"Select atom(s) to modify:",
            options=atom_positions,
            format_func=lambda x: f"{atomic_symbols[x-1]}{x}",
            key=f"atoms_{i}",
        )
        
        group = None
        if mod_type in ["Substitution", "Addition"]:
            group_category = st.sidebar.selectbox(
                "Select functional group category:",
                options=list(groups.keys()),
                key=f"group_category_{i}",
            )
            selected_group = st.sidebar.selectbox(
                f"Select group for modification:",
                options=list(groups[group_category].keys()),
                key=f"group_{i}",
            )
            group = groups[group_category][selected_group]
        
        for pos in selected_positions:
            atom_index = pos - 1
            if mod_type == "Deletion":
                 indices_to_delete.append(atom_index)
            else:
                mods_to_apply.append((mod_type, atom_index, group))


    if st.sidebar.button("Perform Modifications"):
        new_atomic_symbols = atomic_symbols.copy()
        new_atomic_coordinates = atomic_coordinates.copy()
        
        # Deletions are complex with index shifting. Handle them first with care.
        if indices_to_delete:
            # Apply deletions first to simplify subsequent index management
            new_atomic_symbols, new_atomic_coordinates = delete_atoms(
                new_atomic_symbols, new_atomic_coordinates, indices_to_delete
            )
            
            # Now, we must adjust the indices for additions/substitutions
            # This is a complex problem. A simpler approach is to show the result
            # and ask the user to re-run for more modifications.
            # For this implementation, we will process deletions and then stop.
            st.warning("Deletions performed. Please re-run for further additions or substitutions on the new structure.")
            st.session_state['atomic_symbols'] = new_atomic_symbols
            st.session_state['atomic_coordinates'] = new_atomic_coordinates
            st.rerun()

        else: # No deletions, proceed with additions/substitutions
            mods_to_apply.sort(key=lambda x: x[1], reverse=True)

            for mod_type, position, group_data in mods_to_apply:
                if mod_type == "Substitution":
                    new_atomic_symbols, new_atomic_coordinates = replace_atom_with_group(
                        new_atomic_symbols,
                        new_atomic_coordinates,
                        position,
                        group_data,
                    )
                elif mod_type == "Addition":
                    new_atomic_symbols, new_atomic_coordinates = add_group_to_atom(
                        new_atomic_symbols,
                        new_atomic_coordinates,
                        position,
                        group_data,
                    )
            
            # Display modified structure
            st.subheader("Modified Molecule Structure")
            xyz_string_mod = create_xyz_string(new_atomic_symbols, new_atomic_coordinates)

            view_mod = py3Dmol.view(width=800, height=400)
            view_mod.addModel(xyz_string_mod, "xyz")
            view_mod.setStyle({'sphere': {'radius': 0.3}, 'stick': {'radius': 0.15}})
            
            for i, (symbol, coords) in enumerate(zip(new_atomic_symbols, new_atomic_coordinates)):
                view_mod.addLabel(
                    f"{i+1}",
                    {
                        "position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                        "fontSize": 12, "fontColor": "black", "backgroundColor": "white", "backgroundOpacity": 0.5
                    },
                )

            view_mod.zoomTo()
            showmol(view_mod, height=400, width=800)

            modified_xyz = write_xyz(new_atomic_symbols, new_atomic_coordinates)
            st.download_button(
                label="Download Modified XYZ File",
                data=modified_xyz,
                file_name="modified_molecule.xyz",
                mime="text/plain",
            )
