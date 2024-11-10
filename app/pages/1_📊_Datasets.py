# st/page/datasets.py

import streamlit as st
import pandas as pd
import pickle
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact

st.set_page_config(page_title="Dataset Management", page_icon="📂")

# Initialize AutoML system instance
automl_system = AutoMLSystem.get_instance()

def main():
    st.title("Dataset Management")
    st.write("Manage datasets for use in your ML models.")

    # Section 1: Upload and Convert Dataset (Create)
    st.header("Upload and Convert Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Dataset:", df.head())

        # Convert to Dataset and Store Temporarily
        dataset_name = st.text_input("Enter a name for the dataset")
        if st.button("Convert and Save Temporarily"):
            dataset = Dataset.from_dataframe(df)
            st.session_state.setdefault("converted_datasets", {})
            st.session_state["converted_datasets"][dataset_name] = dataset
            st.success(f"Dataset '{dataset_name}' converted and stored temporarily.")

    # Section 2: Save Converted Dataset (Save)
    st.header("Save Converted Dataset")
    if "converted_datasets" in st.session_state and st.session_state["converted_datasets"]:
        dataset_name = st.selectbox("Select a dataset to save", options=st.session_state["converted_datasets"].keys())
        if st.button("Save Dataset"):
            dataset = st.session_state["converted_datasets"][dataset_name]
            artifact = Artifact(name=dataset_name, data=pickle.dumps(dataset))
            automl_system.registry.save(artifact)
            st.success(f"Dataset '{dataset_name}' saved successfully!")
            del st.session_state["converted_datasets"][dataset_name]  # Remove after saving
    else:
        st.write("No converted datasets to save. Please upload and convert a dataset first.")

    # Section 3: View Existing Datasets (List)
    st.header("View Existing Datasets")
    datasets = automl_system.registry.list(type="dataset")
    dataset_names = [artifact.name for artifact in datasets]
    selected_dataset_name = st.selectbox("Choose a dataset to view", options=dataset_names)

    if selected_dataset_name:
        artifact = automl_system.registry.load(selected_dataset_name)
        dataset = pickle.loads(artifact.data)
        st.write("Dataset Preview:")
        st.write(dataset.to_dataframe())  # Assuming Dataset has a to_dataframe method

if __name__ == "__main__":
    main()
