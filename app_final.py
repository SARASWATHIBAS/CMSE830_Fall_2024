# Core imports
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap.umap_ import UMAP
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           r2_score, mean_absolute_error, mean_squared_error, silhouette_score)
from sklearn.model_selection import train_test_split, learning_curve
from xgboost import XGBClassifier, XGBRegressor
from scipy import stats
from imblearn.over_sampling import SMOTE
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.stButton > button {
    background-color: #007BFF;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 10px;
    font-size: 16px;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    overflow-x: scroll;
}
.stTabs [data-baseweb="tab"] {
    min-width: 200px;
    font-size: 14px;
    padding: 5px 10px;
    background-color: #f0f2f6;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
st.session_state.page_views += 1

# Data loading function
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/main/SEER%20Breast%20Cancer%20Dataset%20.csv"
    data = pd.read_csv(url)
    return data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Load data
try:
    data = load_data()
    data_actual = data.copy()
except Exception as e:
    st.error(f"Error loading data: {e}")
# Sidebar components
with st.sidebar:
    st.header("Dashboard Controls")

    # Theme selector
    theme = st.selectbox("Select Theme", ["Light", "Dark"])

    # User guide
    with st.expander("📚 User Guide", expanded=True):
        st.markdown("""
        ### How to Use This Dashboard
        1. **Data Selection**
           - Use filters below to select features
           - Choose categorical and numerical columns
        2. **Analysis Tools**
           - Data Overview: Basic statistics
           - Search: Find specific records
           - Correlation: Analyze relationships
        3. **Visualizations**
           - Interactive plots
           - Custom chart builder
        4. **Feature Engineering**
           - Create new features
           - Transform existing ones
        """)

    # Data filters
    st.header("Filter Data")
    categorical_filter = data.select_dtypes(include='object').columns.tolist()
    numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

    # Default selections
    default_categorical = categorical_filter[:2]
    default_numeric = numeric_filter[:2]

    # Filter selections
    selected_categorical = st.multiselect(
        "Select Categorical Columns",
        categorical_filter,
        default=default_categorical,
        key="categorical_multiselect"
    )

    selected_numeric = st.multiselect(
        "Select Numeric Columns",
        numeric_filter,
        default=default_numeric,
        key="numeric_multiselect"
    )

    # Reset filters button
    if st.button("Reset Filters", key="reset_filters_button"):
        st.session_state.selected_categorical = []
        st.session_state.selected_numeric = []
        st.session_state.is_filtered = False

    # Session tracking
    st.metric("Session Views", st.session_state.page_views)

    # Data refresh button
    if st.button("🔄 Refresh Data"):
        with st.spinner("Refreshing data..."):
            time.sleep(1)
            st.success("Data refreshed!")

# Main content header
st.title("Breast Cancer Analysis Dashboard")
st.write("Interactive analysis and visualization platform for breast cancer data")

# Create main tabs
tabs = st.tabs([
    "📊 Overview",
    "🔍 Search",
    "🔗 Correlation",
    "🧮 Imputation",
    "📈 Scaling",
    "📊 Visualization",
    "🤖 Modeling",
    "🧹 Data Cleaning",
    "📑 Analysis",
    "⚙️ Feature Engineering"
])
# Tab 1: Overview
with tabs[0]:
    st.header("Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write("### Dataset Shape")
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

        st.write("### Sample Data")
        st.dataframe(data.head())

    with col2:
        st.write("### Data Types")
        st.dataframe(pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Missing Values': data.isnull().sum()
        }))

    # Basic statistics
    if selected_numeric:
        st.write("### Numerical Data Statistics")
        st.write(data[selected_numeric].describe())

    if selected_categorical:
        st.write("### Categorical Data Statistics")
        for col in selected_categorical:
            st.write(f"#### {col} Distribution")
            fig = px.pie(data_frame=data[col].value_counts().reset_index(),
                         values=col, names='index', title=f'{col} Distribution')
            st.plotly_chart(fig)

# Tab 2: Search
with tabs[1]:
    st.header("Search Records")

    search_col = st.selectbox("Select Column to Search", data.columns)

    if search_col in categorical_filter:
        search_value = st.selectbox("Select Value", data[search_col].unique())
    else:
        search_value = st.text_input("Enter Search Value")

    if st.button("Search"):
        filtered_data = data[data[search_col].astype(str).str.contains(str(search_value), case=False, na=False)]

        if not filtered_data.empty:
            st.write(f"### Results for: {search_value} in {search_col}")
            st.dataframe(filtered_data)

            st.write("### Summary Statistics")
            st.write(filtered_data.describe(include='all'))
        else:
            st.warning("No results found.")

# Tab 3: Correlation
with tabs[2]:
    st.header("Correlation Analysis")

    if selected_numeric:
        # Correlation matrix
        corr_matrix = data[selected_numeric].corr()

        # Heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig)

        # Detailed correlation analysis
        if st.checkbox("Show Detailed Correlation Analysis"):
            for i in range(len(selected_numeric)):
                for j in range(i + 1, len(selected_numeric)):
                    col1, col2 = selected_numeric[i], selected_numeric[j]
                    correlation = corr_matrix.loc[col1, col2]

                    st.write(f"### {col1} vs {col2}")
                    st.write(f"Correlation: {correlation:.3f}")

                    fig = px.scatter(data, x=col1, y=col2,
                                     title=f"Scatter Plot: {col1} vs {col2}")
                    st.plotly_chart(fig)
    else:
        st.write("Please select numeric columns for correlation analysis.")
# Tab 4: Imputation
with tabs[3]:
    st.header("Data Imputation")

    if selected_numeric:
        imputation_method = st.selectbox(
            "Select Imputation Method",
            ["Mean", "Median", "KNN"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Original Data Statistics")
            st.write(data[selected_numeric].describe())

            # Original data distribution
            for col in selected_numeric:
                fig = px.histogram(data, x=col, title=f"Original Distribution: {col}")
                st.plotly_chart(fig)

        with col2:
            if imputation_method == "Mean":
                imputer = SimpleImputer(strategy='mean')
            elif imputation_method == "Median":
                imputer = SimpleImputer(strategy='median')
            else:
                imputer = KNNImputer(n_neighbors=5)

            imputed_data = pd.DataFrame(
                imputer.fit_transform(data[selected_numeric]),
                columns=selected_numeric
            )

            st.write("### Imputed Data Statistics")
            st.write(imputed_data.describe())

            # Imputed data distribution
            for col in selected_numeric:
                fig = px.histogram(imputed_data, x=col, title=f"Imputed Distribution: {col}")
                st.plotly_chart(fig)

# Tab 5: Scaling
with tabs[4]:
    st.header("Data Scaling")

    if selected_numeric:
        scaling_method = st.selectbox(
            "Select Scaling Method",
            ["StandardScaler", "MinMaxScaler", "RobustScaler"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.write("### Original Data")
            st.write(data[selected_numeric].describe())

            # Original data distribution
            for col in selected_numeric:
                fig = px.box(data, y=col, title=f"Original Distribution: {col}")
                st.plotly_chart(fig)

        with col2:
            if scaling_method == "StandardScaler":
                scaler = StandardScaler()
            elif scaling_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            scaled_data = pd.DataFrame(
                scaler.fit_transform(data[selected_numeric]),
                columns=selected_numeric
            )

            st.write("### Scaled Data")
            st.write(scaled_data.describe())

            # Scaled data distribution
            for col in selected_numeric:
                fig = px.box(scaled_data, y=col, title=f"Scaled Distribution: {col}")
                st.plotly_chart(fig)

# Tab 6: Visualization
with tabs[5]:
    st.header("Advanced Visualizations")

    plot_type = st.selectbox(
        "Select Plot Type",
        ["Scatter", "Box", "Violin", "Bar", "Line", "3D Scatter"]
    )

    if plot_type == "Scatter":
        x_col = st.selectbox("Select X-axis", selected_numeric, key='scatter_x')
        y_col = st.selectbox("Select Y-axis", selected_numeric, key='scatter_y')
        color_col = st.selectbox("Select Color Variable", selected_categorical)

        fig = px.scatter(data, x=x_col, y=y_col, color=color_col,
                         title=f"Scatter Plot: {x_col} vs {y_col}")
        st.plotly_chart(fig)

    elif plot_type == "Box":
        y_col = st.selectbox("Select Numeric Variable", selected_numeric)
        x_col = st.selectbox("Select Categorical Variable", selected_categorical)

        fig = px.box(data, x=x_col, y=y_col,
                     title=f"Box Plot: {y_col} by {x_col}")
        st.plotly_chart(fig)
# Tab 7: Modeling
with tabs[6]:
    st.header("Model Development & Evaluation")

    model_type = st.selectbox(
        "Select Model Type",
        ["Classification", "Regression", "Clustering"]
    )

    if model_type == "Classification":
        # Prepare data
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['Status'])
        X = data[selected_numeric]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        classifier = st.selectbox(
            "Select Classifier",
            ["Random Forest", "XGBoost", "Logistic Regression"]
        )

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if classifier == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif classifier == "XGBoost":
                    model = XGBClassifier(random_state=42)
                else:
                    model = LogisticRegression(random_state=42)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### Model Performance")
                metrics = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted')
                }

                st.write(pd.DataFrame([metrics]))

                if classifier == "Random Forest":
                    st.write("### Feature Importance")
                    importance_df = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)

                    fig = px.bar(importance_df, x='Feature', y='Importance',
                                 title='Feature Importance')
                    st.plotly_chart(fig)

    elif model_type == "Regression":
        # Similar structure for regression models
        y = data['Survival Months']
        X = data[selected_numeric]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        regressor = st.selectbox(
            "Select Regressor",
            ["Random Forest", "XGBoost", "Linear Regression"]
        )

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                if regressor == "Random Forest":
                    model = RandomForestRegressor(random_state=42)
                elif regressor == "XGBoost":
                    model = XGBRegressor(random_state=42)
                else:
                    model = LinearRegression()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("### Model Performance")
                metrics = {
                    'R2 Score': r2_score(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred)
                }

                st.write(pd.DataFrame([metrics]))

    else:  # Clustering
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        if st.button("Perform Clustering"):
            with st.spinner("Clustering data..."):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(StandardScaler().fit_transform(X))

                st.write("### Clustering Results")
                fig = px.scatter(X, x=X.columns[0], y=X.columns[1],
                                 color=clusters,
                                 title='K-Means Clustering Results')
                st.plotly_chart(fig)
# Tab 8: Advanced Data Cleaning
with tabs[7]:
    st.header("Advanced Data Cleaning")

    # Outlier Detection and Removal
    st.subheader("Outlier Detection")
    outlier_method = st.selectbox(
        "Select Outlier Detection Method",
        ["Z-Score", "IQR"]
    )

    if st.button("Detect Outliers"):
        if outlier_method == "Z-Score":
            z_scores = np.abs(stats.zscore(data[selected_numeric]))
            outliers = (z_scores > 3).any(axis=1)
        else:
            Q1 = data[selected_numeric].quantile(0.25)
            Q3 = data[selected_numeric].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data[selected_numeric] < (Q1 - 1.5 * IQR)) |
                        (data[selected_numeric] > (Q3 + 1.5 * IQR))).any(axis=1)

        st.write(f"Number of outliers detected: {outliers.sum()}")

        # Visualize outliers
        for col in selected_numeric:
            fig = px.box(data, y=col, title=f"Outliers in {col}")
            st.plotly_chart(fig)

# Tab 9: Advanced Analysis
with tabs[8]:
    st.header("Advanced Analysis")

    # Dimensionality Reduction
    st.subheader("Dimensionality Reduction")
    dim_reduction = st.selectbox(
        "Select Method",
        ["PCA", "t-SNE", "UMAP"]
    )

    if st.button("Perform Dimensionality Reduction"):
        with st.spinner("Processing..."):
            scaled_data = StandardScaler().fit_transform(data[selected_numeric])

            if dim_reduction == "PCA":
                reducer = PCA(n_components=2)
            elif dim_reduction == "t-SNE":
                reducer = TSNE(n_components=2, random_state=42)
            else:
                reducer = UMAP(n_components=2, random_state=42)

            reduced_data = reducer.fit_transform(scaled_data)

            fig = px.scatter(
                x=reduced_data[:, 0],
                y=reduced_data[:, 1],
                color=data[selected_categorical[0]] if selected_categorical else None,
                title=f"{dim_reduction} Visualization"
            )
            st.plotly_chart(fig)

# Tab 10: Feature Engineering
with tabs[9]:
    st.header("Feature Engineering")

    # Polynomial Features
    st.subheader("Create Polynomial Features")
    degree = st.slider("Select polynomial degree", 2, 5, 2)

    if st.button("Generate Polynomial Features"):
        poly = PolynomialFeatures(degree=degree)
        poly_features = poly.fit_transform(data[selected_numeric])
        feature_names = poly.get_feature_names_out(selected_numeric)

        st.write("### New Polynomial Features")
        st.write(pd.DataFrame(poly_features, columns=feature_names).head())

    # Feature Interactions
    st.subheader("Create Feature Interactions")
    if len(selected_numeric) >= 2:
        feature1 = st.selectbox("Select first feature", selected_numeric)
        feature2 = st.selectbox("Select second feature",
                                [f for f in selected_numeric if f != feature1])

        if st.button("Create Interaction"):
            interaction = data[feature1] * data[feature2]
            st.write(f"### Interaction between {feature1} and {feature2}")
            fig = px.histogram(interaction, title="Interaction Distribution")
            st.plotly_chart(fig)

# Add download button for processed data
st.sidebar.download_button(
    label="Download Processed Data",
    data=data.to_csv(index=False).encode('utf-8'),
    file_name='processed_data.csv',
    mime='text/csv'
)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Name | Last updated: " + datetime.now().strftime("%Y-%m-%d"))
