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

st.title("Structural Modification of Molecules")

# File upload in the main area
uploaded_file = st.file_uploader("Upload XYZ File", type="xyz")

if uploaded_file is not None:
    # Read the XYZ file
    atomic_symbols, atomic_coordinates = read_xyz(uploaded_file)

    # Display original structure with annotations
    st.subheader("Original Molecule Structure")
    xyz_string = create_xyz_string(atomic_symbols, atomic_coordinates)

    view = py3Dmol.view(width=800, height=400)
    view.addModel(xyz_string, "xyz")
    view.setStyle({
        'sphere': {'radius': 0.5},  
        'stick': {'radius': 0.3}   
    })
    # Add labels to atoms
    for i, (symbol, coords) in enumerate(zip(atomic_symbols, atomic_coordinates)):
        view.addLabel(
            f"{i+1}",
            {
                "position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                "fontSize": 14,
                "fontColor": "black",
                "backgroundOpacity": 0.2,
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
            "Propyl (-CH2CH2CH3)": ["C", "H", "H", "C", "H", "H", "C", "H", "H", "H"],
        },
        "Oxygen-containing Groups": {
            "Hydroxyl (-OH)": ["O", "H"],
            "Carbonyl (=O)": ["O"],
            "Carboxyl (-COOH)": ["C", "O", "O", "H"],
            "Ether (-OR)": ["O", "C", "H", "H", "H"],
            "Ester (-COOR)": ["C", "O", "O", "C", "H", "H", "H"],
        },
        "Nitrogen-containing Groups": {
            "Amino (-NH2)": ["N", "H", "H"],
            "Amide (-CONH2)": ["C", "O", "N", "H", "H"],
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
            "Sulfone (-SO2-)": ["S", "O", "O"],
        },
        "Phosphorus-containing Groups": {
            "Phosphate (-PO4)": ["P", "O", "O", "O", "O"],
        },
    }

    modifications = []
    add_another = True
    modification_count = 0

    while add_another:
        st.sidebar.subheader(f"Modification {modification_count + 1}")

        # Choose modification type
        mod_type = st.sidebar.radio(
            "Modification type:",
            ["Substitution", "Addition", "Deletion"],
            key=f"mod_type_{modification_count}",
        )

        # Atom selection
        atom_positions = list(range(1, len(atomic_symbols) + 1))
        selected_positions = st.sidebar.multiselect(
            f"Select atom(s) to modify:",
            options=atom_positions,
            format_func=lambda x: f"{atomic_symbols[x-1]} atom at position {x}",
            key=f"atoms_{modification_count}",
        )

        # Group selection (only for Substitution and Addition)
        if mod_type in ["Substitution", "Addition"]:
            group_category = st.sidebar.selectbox(
                "Select functional group category:",
                options=list(groups.keys()),
                key=f"group_category_{modification_count}",
            )
            selected_group = st.sidebar.selectbox(
                f"Select group for modification:",
                options=list(groups[group_category].keys()),
                key=f"group_{modification_count}",
            )
            group = groups[group_category][selected_group]
        else:
            group = None

        for position in selected_positions:
            modifications.append((mod_type, position - 1, group))

        add_another = st.sidebar.checkbox(
            "Add another modification", key=f"add_{modification_count}"
        )
        modification_count += 1

    if st.sidebar.button("Perform Modifications"):
        # Perform modifications
        new_atomic_symbols = atomic_symbols.copy()
        new_atomic_coordinates = atomic_coordinates.copy()

        # Sort modifications to handle deletions last
        modifications.sort(key=lambda x: (x[0] != "Deletion", x[1]), reverse=True)

        for mod_type, position, group in modifications:
            if mod_type == "Substitution":
                new_atomic_symbols, new_atomic_coordinates = replace_atom_with_group(
                    new_atomic_symbols,
                    new_atomic_coordinates,
                    position,
                    group,
                )
            elif mod_type == "Addition":
                new_atomic_symbols, new_atomic_coordinates = add_group_to_atom(
                    new_atomic_symbols,
                    new_atomic_coordinates,
                    position,
                    group,
                )
            elif mod_type == "Deletion":
                new_atomic_symbols, new_atomic_coordinates = delete_atoms(
                    new_atomic_symbols, new_atomic_coordinates, [position]
                )

        # Display modified structure with annotations
        st.subheader("Modified Molecule Structure")
        xyz_string = create_xyz_string(new_atomic_symbols, new_atomic_coordinates)

        view = py3Dmol.view(width=800, height=400)
        view.addModel(xyz_string, "xyz")
        view.setStyle({"stick": {}})

        # Add labels to atoms
        for i, (symbol, coords) in enumerate(
            zip(new_atomic_symbols, new_atomic_coordinates)
        ):
            view.addLabel(
                f"{i+1}",
                {
                    "position": {"x": coords[0], "y": coords[1], "z": coords[2]},
                    "fontSize": 14,
                    "fontColor": "black",
                    "backgroundOpacity": 0.2,
                },
            )

        view.zoomTo()
        showmol(view, height=400, width=800)

        # Generate modified XYZ file for download
        modified_xyz = write_xyz(new_atomic_symbols, new_atomic_coordinates)
        st.download_button(
            label="Download Modified XYZ File",
            data=modified_xyz,
            file_name="modified_molecule.xyz",
            mime="text/plain",
        )
