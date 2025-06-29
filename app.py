# app.py
import streamlit as st
from stmol import showmol
import py3Dmol
from helper_functions import (
    read_xyz,
    write_xyz,
    replace_atom_with_group,
    add_group_to_atom,
    delete_atoms,
    create_xyz_string,
)

st.set_page_config(layout="wide")
st.title("Structural Modification of Molecules with Hybridization Rules")

# File upload in the main area
uploaded_file = st.file_uploader("Upload XYZ File", type="xyz")

if uploaded_file is not None:
    # Read the XYZ file
    try:
        atomic_symbols, atomic_coordinates = read_xyz(uploaded_file)
        st.session_state['atomic_symbols'] = atomic_symbols
        st.session_state['atomic_coordinates'] = atomic_coordinates
    except ValueError as e:
        st.error(e)
        st.stop()


if 'atomic_symbols' in st.session_state:
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

    # Expanded list of functional groups
    groups = {
        "Alkyl Groups": {
            "Methyl (-CH3)": ["C", "H", "H", "H"],
            "Ethyl (-CH2CH3)": ["C", "H", "H", "C", "H", "H", "H"],
        },
        "Oxygen-containing Groups": {
            "Hydroxyl (-OH)": ["O", "H"],
            "Carbonyl (=O)": ["O"],
            "Carboxyl (-COOH)": ["C", "O", "O", "H"],
        },
        "Nitrogen-containing Groups": {
            "Amino (-NH2)": ["N", "H", "H"],
            "Nitro (-NO2)": ["N", "O", "O"],
        },
        "Halogen Groups": {
            "Fluoro (-F)": ["F"],
            "Chloro (-Cl)": ["Cl"],
            "Bromo (-Br)": ["Br"],
            "Iodo (-I)": ["I"],
        },
        "Sulfur-containing Groups": {
            "Thiol (-SH)": ["S", "H"],
        },
    }

    # Using session state to manage modifications
    if 'modifications' not in st.session_state:
        st.session_state.modifications = []

    def add_modification():
        st.session_state.modifications.append({'type': 'Addition', 'atoms': [], 'group_cat': 'Alkyl Groups', 'group_sel': 'Methyl (-CH3)'})

    st.sidebar.button("Add another modification", on_click=add_modification)

    # Store modifications to apply
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
            # Adjust for 0-based indexing
            atom_index = pos - 1
            if mod_type == "Deletion":
                 indices_to_delete.append(atom_index)
            else:
                mods_to_apply.append((mod_type, atom_index, group))


    if st.sidebar.button("Perform Modifications"):
        # Apply additions and substitutions first
        new_atomic_symbols = atomic_symbols.copy()
        new_atomic_coordinates = atomic_coordinates.copy()

        # Sort modifications to handle indices correctly
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
        
        # Apply deletions after all additions/substitutions are done
        if indices_to_delete:
            new_atomic_symbols, new_atomic_coordinates = delete_atoms(
                new_atomic_symbols, new_atomic_coordinates, indices_to_delete
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

        # Generate modified XYZ file for download
        modified_xyz = write_xyz(new_atomic_symbols, new_atomic_coordinates)
        st.download_button(
            label="Download Modified XYZ File",
            data=modified_xyz,
            file_name="modified_molecule.xyz",
            mime="text/plain",
        )
