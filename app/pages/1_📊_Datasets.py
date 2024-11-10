import streamlit as st
import pandas as pd
import os
import json

# Title for the dataset management page
st.title("Dataset Management")

# Directories and files
ASSETS_DIR = 'assets'
OBJECTS_DIR = os.path.join(ASSETS_DIR, 'objects')
REGISTRY_FILE = os.path.join(ASSETS_DIR, 'registry.json')

# Ensure directories exist
os.makedirs(OBJECTS_DIR, exist_ok=True)


# Function to load the registry
def load_registry() -> list:
    """Load registry from file if exists, else return empty list."""
    try:
        if os.path.exists(REGISTRY_FILE):
            with open(REGISTRY_FILE, 'r') as f:
                registry = json.load(f)
                if isinstance(registry, list):
                    return registry
    except json.JSONDecodeError:
        st.error("Error reading the registry file. Reinitializing registry.")
    return []  # Return an empty list if file is missing or corrupted


# Function to save the registry
def save_registry(registry: list) -> None:
    """Save the registry to a file."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=4)


# Function to refresh dataset list
def refresh_datasets() -> list:
    """Returns a list of dataset file names from the registry."""
    registry = load_registry()
    return [entry['name'] for entry in registry if entry['type'] == 'dataset']


# Initialize dataset list in session state
if 'dataset_files' not in st.session_state:
    st.session_state['dataset_files'] = refresh_datasets()

# Section: List Available Datasets
st.subheader("Available Datasets")

if st.session_state['dataset_files']:
    dataset_options = st.session_state['dataset_files']

    # Selectbox for dataset selection
    selected_dataset_name = st.selectbox(
        "Select a dataset to view or delete",
        options=dataset_options
    )

    # Load and display the selected dataset
    if selected_dataset_name:
        dataset_path = os.path.join(
            OBJECTS_DIR, f"{selected_dataset_name}.csv"
        )
        if os.path.exists(dataset_path):
            dataset_df = pd.read_csv(dataset_path)
            st.write(f"*Dataset Name:* {selected_dataset_name}")
            st.write("*Sample Data:*")
            st.dataframe(dataset_df.head())
        else:
            st.error(f"Dataset '{selected_dataset_name}' not found.")
            st.session_state['dataset_files'] = refresh_datasets()

    # Delete the selected dataset
    if st.button("Delete Dataset"):
        if selected_dataset_name:
            dataset_path = os.path.join(
                OBJECTS_DIR, f"{selected_dataset_name}.csv"
            )
            if os.path.exists(dataset_path):
                os.remove(dataset_path)
                st.success(f"Deleted dataset: {selected_dataset_name}")

                # Update the registry
                registry = load_registry()
                registry = [
                    e for e in registry if e['name'] != selected_dataset_name
                ]
                save_registry(registry)

                # Refresh dataset list
                st.session_state['dataset_files'] = refresh_datasets()
                # Set a flag in session state to trigger a re-render
                st.session_state['refresh_flag'] = not st.session_state.get(
                    'refresh_flag', False
                )
else:
    st.write("No datasets available.")

# Section: Upload New Dataset
st.subheader("Upload New Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# If a file is uploaded, show preview and save options
if uploaded_file:
    # Load CSV into a DataFrame and display a preview
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    # Input for dataset name
    dataset_name = st.text_input("Enter a name for the dataset")

    # Function to save the dataset
    if st.button("Save Dataset"):
        if dataset_name:
            dataset_path = os.path.join(OBJECTS_DIR, f"{dataset_name}.csv")
            if os.path.exists(dataset_path):
                st.error(
                    f"A dataset with the name '{dataset_name}' already exists."
                )
            else:
                # Save CSV and update registry
                df.to_csv(dataset_path, index=False)
                st.success(
                    f"Dataset '{dataset_name}' has been saved successfully."
                )

                # Update the registry
                registry = load_registry()
                new_entry = {
                    "name": dataset_name,
                    "type": "dataset",
                    "asset_path": os.path.relpath(
                        dataset_path, start=ASSETS_DIR
                    )
                }
                registry.append(new_entry)
                save_registry(registry)

                # Refresh dataset list in session state
                st.session_state['dataset_files'] = refresh_datasets()
                # Set a flag in session state to trigger a re-render
                st.session_state['refresh_flag'] = not st.session_state.get(
                    'refresh_flag', False
                    )
        else:
            st.error("Please enter a name for the dataset.")
