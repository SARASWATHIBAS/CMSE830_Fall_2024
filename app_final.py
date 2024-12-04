"""
Breast Cancer Analysis Application
A comprehensive data science tool for analyzing breast cancer data through:
1. Clinical Analysis: Risk assessment and survival predictions
2. Research Insights: Statistical analysis and machine learning
3. Interactive Learning: Dynamic visualizations and explanations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import QuantileTransformer, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           r2_score, mean_absolute_error, mean_squared_error, silhouette_score)
from sklearn.model_selection import learning_curve, train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.impute import KNNImputer, SimpleImputer
import numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans

# Data loading and caching
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/main/SEER%20Breast%20Cancer%20Dataset%20.csv"
    return pd.read_csv(url)

# Set page configuration
st.set_page_config(page_title="Breast Cancer Analysis", layout="wide")


# Core utility functions
def show_documentation():
    """Display comprehensive documentation and user guide"""
    with st.expander("📚 Documentation & User Guide", expanded=False):
        st.markdown("""
        # Breast Cancer Analysis App Documentation

        ## Key Features
        1. **Clinical Analysis**
           - Risk assessment
           - Survival prediction
           - Treatment recommendations

        2. **Research Tools**
           - Statistical analysis
           - Machine learning models
           - Feature importance

        3. **Data Processing**
           - Advanced cleaning
           - Feature engineering
           - Dimensionality reduction

        ## Workflow Guide
        1. Start with Data Overview
        2. Explore visualizations
        3. Apply preprocessing
        4. Build and evaluate models
        """)


# Data preprocessing functions
@st.cache_data
def process_features(data):
    """Process and categorize dataset features"""
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
    return numeric_cols, categorical_cols


def clean_data(data):
    """Perform initial data cleaning"""
    # Remove unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    # Handle missing values
    numeric_imputer = SimpleImputer(strategy='mean')
    categorical_imputer = SimpleImputer(strategy='most_frequent')

    numeric_cols, categorical_cols = process_features(data)

    data[numeric_cols] = numeric_imputer.fit_transform(data[numeric_cols])
    data[categorical_cols] = categorical_imputer.fit_transform(data[categorical_cols])

    return data


def scale_features(data, method='standard'):
    """Scale numeric features using specified method"""
    scalers = {
        'standard': StandardScaler(),
        'minmax': MinMaxScaler(),
        'robust': RobustScaler()
    }
    scaler = scalers.get(method, StandardScaler())
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data


# Load and prepare initial data
data = load_data()
data = clean_data(data)
numeric_cols, categorical_cols = process_features(data)


# Main app layout and sidebar
def create_sidebar():
    st.sidebar.header("Analysis Controls")

    # Feature selection
    st.sidebar.subheader("Select Features")
    selected_numeric = st.sidebar.multiselect(
        "Numeric Features",
        numeric_cols,
        default=numeric_cols[:2]
    )
    selected_categorical = st.sidebar.multiselect(
        "Categorical Features",
        categorical_cols,
        default=categorical_cols[:1]
    )

    # Analysis type
    analysis_type = st.sidebar.selectbox(
        "Analysis Type",
        ["Clinical", "Research", "Educational"]
    )

    return selected_numeric, selected_categorical, analysis_type


def main():
    st.title("Breast Cancer Analysis Platform")
    show_documentation()

    # Create sidebar controls
    selected_numeric, selected_categorical, analysis_type = create_sidebar()

    # Create main tabs
    tabs = st.tabs([
        "Overview",
        "Data Analysis",
        "Visualization",
        "Modeling",
        "Advanced Analysis"
    ])

    # Tab content
    with tabs[0]:  # Overview
        st.header("Data Overview")
        st.write("### Dataset Statistics")
        st.write(data[selected_numeric + selected_categorical].describe())

        if st.checkbox("Show Data Quality Report"):
            st.write("### Data Quality Analysis")
            missing_data = data.isnull().sum()
            st.write("Missing Values:", missing_data[missing_data > 0])

    with tabs[1]:  # Data Analysis
        st.header("Statistical Analysis")

        # Correlation analysis
        if len(selected_numeric) > 1:
            st.subheader("Feature Correlations")
            corr_matrix = data[selected_numeric].corr()
            fig = px.imshow(corr_matrix,
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu')
            st.plotly_chart(fig)

        # Distribution analysis
        st.subheader("Feature Distributions")
        selected_feature = st.selectbox("Select Feature", selected_numeric)
        fig = px.histogram(data, x=selected_feature,
                           title=f"Distribution of {selected_feature}")
        st.plotly_chart(fig)
    with tabs[2]:  # Visualization
        st.header("Interactive Visualizations")

        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plot", "Box Plot", "Violin Plot", "3D Scatter"]
        )

        if viz_type == "Scatter Plot":
            x_col = st.selectbox("X-axis", selected_numeric, key='scatter_x')
            y_col = st.selectbox("Y-axis", selected_numeric, key='scatter_y')
            color_col = st.selectbox("Color by", selected_categorical)

            fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                             title=f"{x_col} vs {y_col} by {color_col}")
            st.plotly_chart(fig)

        elif viz_type == "Box Plot":
            numeric_col = st.selectbox("Numeric Feature", selected_numeric)
            category_col = st.selectbox("Categorical Feature", selected_categorical)

            fig = px.box(data, x=category_col, y=numeric_col,
                         title=f"Distribution of {numeric_col} by {category_col}")
            st.plotly_chart(fig)

        elif viz_type == "Violin Plot":
            numeric_col = st.selectbox("Numeric Feature", selected_numeric, key='violin_num')
            category_col = st.selectbox("Categorical Feature", selected_categorical, key='violin_cat')

            fig = px.violin(data, x=category_col, y=numeric_col,
                            title=f"Distribution of {numeric_col} by {category_col}")
            st.plotly_chart(fig)

        else:  # 3D Scatter
            if len(selected_numeric) >= 3:
                x_col = st.selectbox("X-axis", selected_numeric, key='3d_x')
                y_col = st.selectbox("Y-axis", selected_numeric, key='3d_y')
                z_col = st.selectbox("Z-axis", selected_numeric, key='3d_z')
                color_col = st.selectbox("Color by", selected_categorical, key='3d_color')

                fig = px.scatter_3d(data, x=x_col, y=y_col, z=z_col,
                                    color=color_col,
                                    title="3D Feature Visualization")
                st.plotly_chart(fig)
            else:
                st.warning("Please select at least 3 numeric features for 3D visualization")
    with tabs[3]:  # Modeling
        st.header("Machine Learning Models")

        model_type = st.selectbox(
            "Select Model Type",
            ["Classification", "Regression", "Clustering"]
        )

        if model_type in ["Classification", "Regression"]:
            # Prepare data
            X = data[selected_numeric]
            y = data['Status'] if model_type == "Classification" else data['Survival Months']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "Classification":
                models = {
                    'Random Forest': RandomForestClassifier(),
                    'XGBoost': XGBClassifier(),
                    'Logistic Regression': LogisticRegression()
                }

                selected_model = st.selectbox("Select Model", list(models.keys()))
                model = models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Display metrics
                st.write("Model Performance:")
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted')
                }

                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.3f}")

            else:  # Regression
                models = {
                    'Random Forest': RandomForestRegressor(),
                    'XGBoost': XGBRegressor(),
                    'Linear Regression': LinearRegression()
                }

                selected_model = st.selectbox("Select Model", list(models.keys()))
                model = models[selected_model]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Display metrics
                st.write("Model Performance:")
                metrics = {
                    'R2 Score': r2_score(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred)
                }

                for metric, value in metrics.items():
                    st.metric(metric, f"{value:.3f}")

        else:  # Clustering
            n_clusters = st.slider("Number of Clusters", 2, 8, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(data[selected_numeric])

            # Visualize clusters
            if len(selected_numeric) >= 2:
                fig = px.scatter(
                    data, x=selected_numeric[0], y=selected_numeric[1],
                    color=clusters,
                    title="Cluster Visualization"
                )
                st.plotly_chart(fig)

    with tabs[4]:  # Advanced Analysis
        st.header("Advanced Analysis Tools")

        analysis_tool = st.selectbox(
            "Select Analysis Tool",
            ["Dimensionality Reduction", "Feature Importance", "Survival Analysis"]
        )

        if analysis_tool == "Dimensionality Reduction":
            method = st.selectbox(
                "Select Method",
                ["PCA", "t-SNE", "UMAP"]
            )

            # Apply dimensionality reduction
            X = StandardScaler().fit_transform(data[selected_numeric])

            if method == "PCA":
                reducer = PCA(n_components=2)
            elif method == "t-SNE":
                reducer = TSNE(n_components=2)
            else:
                reducer = UMAP(n_components=2)

            reduced_data = reducer.fit_transform(X)

            fig = px.scatter(
                x=reduced_data[:, 0], y=reduced_data[:, 1],
                color=data[selected_categorical[0]] if selected_categorical else None,
                title=f"{method} Visualization"
            )
            st.plotly_chart(fig)


if __name__ == "__main__":
    main()
