import streamlit as st
import pandas as pd
import pickle

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline  # Assuming you have a Pipeline class for modeling
from autoop.core.ml.feature import Feature
from autoop.core.ml.metric import Metric

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

# Initialize AutoML system instance and load datasets
automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

# Step 1: Select Dataset
st.header("Step 1: Select Dataset")
dataset_names = [artifact.name for artifact in datasets]
selected_dataset_name = st.selectbox("Choose a dataset", options=dataset_names)

if selected_dataset_name:
    # Load the selected dataset
    artifact = automl.registry.load(selected_dataset_name)
    dataset = pickle.loads(artifact.data)
    st.write("Dataset Preview:")
    st.write(dataset.to_dataframe())  # Assuming Dataset has a method to return a DataFrame

    # Step 2: Choose Features and Target
    st.header("Step 2: Select Features and Target")
    feature_names = dataset.get_feature_names()  # Assuming Dataset has this method
    selected_features = st.multiselect("Choose input features", options=feature_names)
    target_feature = st.selectbox("Select target feature", options=feature_names)

    if target_feature:
        # Detect task type based on the target feature type
        target_type = dataset.get_feature_type(target_feature)  # Assuming Dataset can provide feature types
        task_type = "classification" if target_type == "categorical" else "regression"
        st.write(f"Detected task type: {task_type}")

        # Step 3: Select Model
        st.header("Step 3: Select Model")
        available_models = automl.get_models(task_type=task_type)  # Get models based on task type
        selected_model_name = st.selectbox("Choose a model", options=available_models.keys())
        selected_model = available_models[selected_model_name] if selected_model_name else None

        # Step 4: Set Train-Test Split
        st.header("Step 4: Set Train-Test Split")
        split_ratio = st.slider("Train-Test Split Ratio", min_value=0.1, max_value=0.9, value=0.8)

        # Step 5: Select Metrics
        st.header("Step 5: Select Evaluation Metrics")
        available_metrics = automl.get_metrics(task_type=task_type)  # Get metrics based on task type
        selected_metrics = st.multiselect("Choose evaluation metrics", options=available_metrics.keys())
        metric_objects = [available_metrics[metric] for metric in selected_metrics]

        # Step 6: Run the Pipeline
        st.header("Step 6: Train and Evaluate Model")
        if st.button("Run Pipeline"):
            # Set up pipeline and train model
            input_features = [Feature(name) for name in selected_features]
            target = Feature(target_feature)
            
            pipeline = Pipeline(
                metrics=metric_objects,
                dataset=dataset,
                model=selected_model,
                input_features=input_features,
                target_feature=target,
                split=split_ratio
            )
            
            # Execute pipeline and display results
            results = pipeline.execute()
            st.write("Pipeline Results:")
            st.write("Training Metrics:", results["train_metrics"])
            st.write("Test Metrics:", results["test_metrics"])
            st.write("Predictions on Test Set:", results["test_predictions"])
