import streamlit as st
import pandas as pd
import io

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f'<p style="color: #888;">{text}</p>', unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a "
    "model on a dataset."
)

# Initialize the AutoML system
automl = AutoMLSystem.get_instance()

# Load existing datasets from the artifact registry
datasets = automl.registry.list(type="dataset")

# **ST/modelling/datasets/list:** Load existing datasets using the artifact
dataset_names = [dataset.name for dataset in datasets]
selected_dataset_name = st.selectbox("Select a Dataset", dataset_names)

# Retrieve the selected dataset
selected_dataset = next(
    (dataset for dataset in datasets if dataset.name == selected_dataset_name),
    None
)

if selected_dataset:
    # Read dataset data into a pandas DataFrame
    try:
        csv_data = selected_dataset.data.decode('utf-8')
        dataset_df = pd.read_csv(io.StringIO(csv_data))
        st.write("## Dataset Preview")
        st.dataframe(dataset_df.head())
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.error("Dataset not found.")
    st.stop()

# Convert DataFrame back to Dataset object
dataset = Dataset.from_dataframe(
    data=dataset_df,
    name=selected_dataset_name,
    asset_path=selected_dataset.asset_path
)

# **ST/modelling/datasets/features:** Detect features and generate selection
try:
    # Detect feature types using your implemented function
    detected_features = detect_feature_types(dataset)
    numerical_features = [
        feature.name for feature in detected_features
        if feature.type == 'numerical'
    ]
    categorical_features = [
        feature.name for feature in detected_features
        if feature.type == 'categorical'
    ]
except Exception as e:
    st.error(f"Error detecting features: {e}")
    st.stop()

all_features = numerical_features + categorical_features

input_features = st.multiselect("Select Input Features", options=all_features)
target_feature = st.selectbox("Select Target Feature", options=all_features)

# Ensure target feature is not in input features
if target_feature in input_features:
    st.warning(
        "Target feature should not be in input features. "
        "Please adjust your selection."
    )
    st.stop()

# **Prompt the user with the detected task type**
if dataset_df[target_feature].dtype == object or dataset_df[
    target_feature
].nunique() < 10:
    task_type = 'classification'
else:
    task_type = 'regression'

st.write(
    f"**Detected task type:** {task_type.capitalize()}"
)

st.write("## Select a Model")

models = []
if task_type == 'classification':
    models = ['Logistic Regression', 'Decision Tree Classifier']
elif task_type == 'regression':
    models = ['Linear Regression', 'Decision Tree Regressor']

selected_model_name = st.selectbox("Select a Model", models)

st.write("## Select Dataset Split")

test_size = st.slider(
    "Test Set Size (%)", min_value=10, max_value=50, value=20, step=5
)
random_state = st.number_input(
    "Random State (for reproducibility)", value=42, step=1
)

st.write("## Select Evaluation Metrics")

if task_type == 'classification':
    metric_options = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
elif task_type == 'regression':
    metric_options = ['Mean Squared Error', 'Mean Absolute Error', 'R2 Score']

selected_metrics = st.multiselect(
    "Select Evaluation Metrics", options=metric_options
)

# **ST/modelling/pipeline/summary:** Provide a pipeline summary
st.write("## Pipeline Summary")

st.write(f"**Dataset:** {selected_dataset_name}")
st.write(f"**Input Features:** {', '.join(input_features)}")
st.write(f"**Target Feature:** {target_feature}")
st.write(f"**Task Type:** {task_type.capitalize()}")
st.write(f"**Model:** {selected_model_name}")
st.write(f"**Test Set Size:** {test_size}%")
st.write(f"**Random State:** {random_state}")
st.write(f"**Evaluation Metrics:** {', '.join(selected_metrics)}")

# **ST/modelling/pipeline/train:** Train the model and report the results
if st.button("Train Model"):
    # Prepare the data
    X = dataset_df[input_features]
    y = dataset_df[target_feature]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)
    if y.dtype == object:
        y = pd.factorize(y)[0]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=random_state
    )

    # Initialize the model
    if selected_model_name == 'Logistic Regression':
        model = LogisticRegression()
    elif selected_model_name == 'Decision Tree Classifier':
        model = DecisionTreeClassifier()
    elif selected_model_name == 'Linear Regression':
        model = LinearRegression()
    elif selected_model_name == 'Decision Tree Regressor':
        model = DecisionTreeRegressor()
    else:
        st.error("Invalid model selected.")
        st.stop()

    # Train the model
    try:
        model.fit(X_train, y_train)
        st.success("Model training completed successfully!")
    except Exception as e:
        st.error(f"Error during model training: {e}")
        st.stop()

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    st.write("## Evaluation Results")

    if task_type == 'classification':
        if 'Accuracy' in selected_metrics:
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Accuracy:** {accuracy:.4f}")
        if 'Precision' in selected_metrics:
            precision = precision_score(
                y_test, y_pred, average='weighted', zero_division=0
            )
            st.write(f"**Precision:** {precision:.4f}")
        if 'Recall' in selected_metrics:
            recall = recall_score(
                y_test, y_pred, average='weighted', zero_division=0
            )
            st.write(f"**Recall:** {recall:.4f}")
        if 'F1 Score' in selected_metrics:
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            st.write(f"**F1 Score:** {f1:.4f}")
    elif task_type == 'regression':
        if 'Mean Squared Error' in selected_metrics:
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"**Mean Squared Error:** {mse:.4f}")
        if 'Mean Absolute Error' in selected_metrics:
            mae = mean_absolute_error(y_test, y_pred)
            st.write(f"**Mean Absolute Error:** {mae:.4f}")
        if 'R2 Score' in selected_metrics:
            r2 = r2_score(y_test, y_pred)
            st.write(f"**R² Score:** {r2:.4f}")
