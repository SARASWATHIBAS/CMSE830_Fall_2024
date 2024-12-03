import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures, QuantileTransformer

# Enhanced page configuration
st.set_page_config(
    page_title="Breast Cancer Analysis Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'page_views' not in st.session_state:
    st.session_state.page_views = 0
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
st.session_state.page_views += 1

# Cache the data loading
@st.cache_data(ttl=3600)
def load_data():
    try:
        url = "https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/main/SEER%20Breast%20Cancer%20Dataset%20.csv"
        data = pd.read_csv(url)
        return data.loc[:, ~data.columns.str.contains('^Unnamed')]
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the data
data = load_data()
if data is None:
    st.stop()

# Store original data
data_actual = data.copy()


# Interactive User Guide
def show_user_guide():
    with st.expander("📚 Interactive User Guide", expanded=True):
        st.markdown("""
        ### Welcome to the Breast Cancer Analysis Dashboard! 

        #### 🎯 Key Features:
        1. **Data Analysis**
           - Filter and explore breast cancer data
           - Visualize patterns and trends
           - Perform statistical analysis

        2. **Machine Learning**
           - Train predictive models
           - Evaluate model performance
           - Feature importance analysis

        3. **Data Processing**
           - Handle missing values
           - Scale and normalize data
           - Feature engineering

        #### 🔍 Quick Start:
        1. Select features in the sidebar
        2. Choose analysis methods from tabs
        3. Interact with visualizations
        4. Export results
        """)


# Enhanced sidebar
st.sidebar.title("Analysis Controls")

# Theme selector
theme = st.sidebar.select_slider(
    "Dashboard Theme",
    options=["Light", "Modern", "Dark"],
    value="Modern"
)

# Feature selection
categorical_filter = data.select_dtypes(include='object').columns.tolist()
numeric_filter = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

selected_categorical = st.sidebar.multiselect(
    "Select Categorical Features",
    categorical_filter,
    default=categorical_filter[:2]
)

selected_numeric = st.sidebar.multiselect(
    "Select Numeric Features",
    numeric_filter,
    default=numeric_filter[:2]
)

# Analysis history tracker
if st.sidebar.button("Save Analysis"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.analysis_history.append({
        'timestamp': timestamp,
        'features': selected_categorical + selected_numeric,
        'theme': theme
    })

# Show analysis history
if st.sidebar.checkbox("Show History"):
    for item in st.session_state.analysis_history:
        st.sidebar.write(f"📅 {item['timestamp']}")
        st.sidebar.write(f"Features: {', '.join(item['features'])}")
        st.sidebar.write("---")

# Reset button
if st.sidebar.button("Reset All"):
    st.session_state.clear()
    st.experimental_rerun()

# Main title and user guide
st.title("Breast Cancer Analysis Dashboard")
show_user_guide()

# Create tabs
tabs = st.tabs([
    "Overview", "Search", "Correlation", "Imputation",
    "Scaling", "Visualization", "Modeling",
    "Data Cleaning", "Analysis", "Feature Engineering"
])

# Data Overview Tab
with tab1:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    # Static Overview
    st.header("Data Overview")

    # Message for Feature Selection
    st.write("### Feature Selection for Analysis")
    st.write("""
    Please select **two numeric features** and **one categorical feature** to begin your analysis.
    - **Numeric Features:** These can include measurements such as Tumor Size or Age.
    - **Categorical Feature:** This could be a classification such as Cancer Stage.

    Once selected, you'll be able to explore relationships, perform visualizations, and gain insights from the data!
    """)
    st.markdown("""
    **Overview of the Breast Cancer Dataset:**

    - **Age:** Patient’s age group.
    - **Race:** Racial classification.
    - **Marital Status:** Current marital status of the patient.
    - **T Stage:** Tumor stage based on size.
    - **N Stage:** Lymph node involvement stage.
    - **6th Stage:** Stage classification according to the 6th edition of AJCC.
    - **Grade:** Tumor grade indicating aggressiveness.
    - **A Stage:** Distant metastasis presence/absence.
    - **Tumor Size:** Size of the tumor in mm.
    - **Estrogen/Progesterone Status:** Hormone receptor status.
    - **reginol Node Examined/Positive:** Number of nodes examined and found positive.
    - **Survival Months:** Months patient survived.
    - **Status:** Patient’s survival status.

    This dataset provides key demographic, clinical and pathological data useful for breast cancer analysis.
    """)

    if selected_categorical:
        st.write("### Selected Categorical Data")
        st.write(data[selected_categorical].describe())
    if selected_numeric:
        st.write("### Selected Numeric Data")
        st.write(data[selected_numeric].describe())

# Search Tab with Dropdown for Categorical Variables
with tab2:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Search Data")

    # Select column to search within
    search_column = st.selectbox("Select Column to Search", data.columns)

    if search_column in categorical_filter:
        # If the selected column is categorical, show dropdown with unique values
        search_value = st.selectbox("Select Value", data[search_column].unique())
    else:
        # If the selected column is numeric, allow manual input
        search_value = st.text_input("Enter Search Value")

        # Enter Button to trigger the search
        if st.button("Go"):
            # Filter data based on the search
            filtered_data = data[data[search_column].astype(str).str.contains(str(search_value), case=False, na=False)]

            if not filtered_data.empty:
                st.write(f"### Results for: **{search_value}** in **{search_column}**")
                st.write(filtered_data)

                # Display basic statistics
                st.write("### Summary Statistics")
                st.write(filtered_data.describe(include='all'))

                # Additional insights for numeric data
                if not filtered_data.select_dtypes(include=np.number).empty:
                    st.write("#### Additional Numeric Statistics")
                    st.write(
                        filtered_data.select_dtypes(include=np.number).agg(['mean', 'median', 'min', 'max', 'count']))
            else:
                st.warning("No results found.")

    st.write("Thank you for using the Breast Cancer Analysis App!")

# Correlation Heatmap Tab
with tab3:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Correlation Heatmap")
    if selected_numeric:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[selected_numeric])
        corr_matrix = np.corrcoef(scaled_data.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=selected_numeric, yticklabels=selected_numeric,
                    cmap='coolwarm')
        st.pyplot(fig)
    else:
        st.write("Please select numeric columns for the correlation heatmap.")

# Data Imputation Comparison Tab
with tab4:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Imputation Methods: Mean vs KNN")
    if selected_numeric:
        # Mean Imputation
        mean_imputer = SimpleImputer(strategy='mean')
        data_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(data[selected_numeric]),
                                         columns=selected_numeric)

        # KNN Imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data[selected_numeric]),
                                        columns=selected_numeric)

        # Show distribution comparisons
        for col in selected_numeric:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(data_mean_imputed[col], kde=True, ax=ax[0], color='skyblue')
            ax[0].set_title(f'Mean Imputed: {col}')

            sns.histplot(data_knn_imputed[col], kde=True, ax=ax[1], color='salmon')
            ax[1].set_title(f'KNN Imputed: {col}')

            st.pyplot(fig)
    else:
        st.write("Please select numeric columns for imputation comparison.")

# Min-Max Scaling Tab
with tab5:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Min-Max Scaling")
    if selected_numeric:
        min_max_scaler = MinMaxScaler()
        scaled_data_min_max = pd.DataFrame(min_max_scaler.fit_transform(data[selected_numeric]),
                                           columns=selected_numeric)

        st.write(scaled_data_min_max.head())
    else:
        st.write("Please select numeric columns for min-max scaling.")

# Advanced Visualizations Tab
with tab6:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Advanced Visualizations")

    # Select features for dynamic plots
    feature_x = st.selectbox("Select Feature X", selected_numeric, key="feature_x")

    # Exclude the selected X feature from Y's options
    remaining_features_y = [col for col in selected_numeric if col != feature_x]
    feature_y = st.selectbox("Select Feature Y", remaining_features_y, key="feature_y")
    # Hue feature selection
    hue_feature = st.selectbox("Select Hue (Categorical)", selected_categorical, key="hue_feature")

    if feature_x and feature_y:
        # Scatter plot for the selected features
        # fig, ax = plt.subplots()
        # sns.scatterplot(data=data, x=feature_x, y=feature_y, hue=data[hue_feature])
        # ax.set_title(f'Scatter Plot: {feature_x} vs {feature_y} with hue {hue_feature}')
        # st.pyplot(fig)

        fig = px.scatter(data, x=feature_x, y=feature_y, color=data[hue_feature], title="Interactive Scatter Plot")
        st.plotly_chart(fig)

        st.write("### Interpretation:")
        st.write("The interactive scatter plot allows users to hover over points for more information.")

        # Calculate and display correlation
        correlation = data[feature_x].corr(data[feature_y])
        st.write(f"### I give you dynamic correlation interpretation")
        st.write(f"### Correlation Coefficient between {feature_x} and {feature_y}: {correlation:.2f}")

        # Interpretation
        if correlation > 0.5:
            st.write("### Interpretation: Strong positive relationship.")
        elif correlation < -0.5:
            st.write("### Interpretation: Strong negative relationship.")
        else:
            st.write("### Interpretation: Weak or no relationship.")

    # Additional visualizations
    if st.checkbox("Show Pair Plot"):
        st.subheader("Pair Plot")
        sns.pairplot(data[selected_numeric + [hue_feature]], hue=hue_feature)
        st.pyplot()

    # Violin Plot for two variables
    st.subheader("Violin Plot")
    numerical_feature = st.selectbox("Select Numerical Feature", numeric_filter)
    categorical_feature = st.selectbox("Select Categorical Feature", categorical_filter)
    hue_feature_plot = st.selectbox("Select Hue feature", selected_categorical)

    show_violin = st.checkbox("Show Violin Plot")

    if show_violin and numerical_feature and categorical_feature:
        fig, ax = plt.subplots()
        sns.violinplot(x=data[categorical_feature], y=data[numerical_feature], hue=data[hue_feature_plot], ax=ax)
        ax.set_title(f'Violin Plot of {numerical_feature} by {categorical_feature} with hue {hue_feature_plot}')
        st.pyplot(fig)

    # Box Plot for two variables
    st.subheader("Box Plot")
    show_box = st.checkbox("Show Box Plot")

    if show_box and numerical_feature and categorical_feature:
        fig, ax = plt.subplots()
        sns.boxplot(x=data[categorical_feature], y=data[numerical_feature], hue=data[hue_feature_plot], ax=ax)
        ax.set_title(f'Box Plot of {numerical_feature} by {categorical_feature} with hue {hue_feature_plot}')
        st.pyplot(fig)

    # Closing message
st.write("### Thank you for using the Breast Cancer Analysis App!")

# Modeling Tab
with tab7:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Model Development & Evaluation")

    # Model Selection Section
    st.subheader("1. Model Selection")
    model_type = st.selectbox(
        "Select Model Type",
        ["Classification", "Clustering", "Regression"]
    )

    if model_type == "Classification":
        # Classification Models
        st.write("### Survival Prediction Models")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(data['Status'])  # This will convert 'Alive'/'Dead' to 0/1

        # Feature Selection
        X = data[['Age', 'Tumor Size', 'Reginol Node Positive']]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1': f1_score(y_test, y_pred, average='weighted')
            }

        # Results Visualization
        results_df = pd.DataFrame(results).T
        fig = px.bar(results_df, barmode='group',
                     title='Model Performance Comparison')
        st.plotly_chart(fig)

        # Feature Importance
        if st.checkbox("Show Feature Importance"):
            rf_model = models['Random Forest']
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': rf_model.feature_importances_
            })
            fig = px.bar(importance_df, x='Feature', y='Importance',
                         title='Feature Importance')
            st.plotly_chart(fig)

    elif model_type == "Clustering":
        st.write("### K-Means Clustering Analysis")

        # Feature Selection for Clustering
        numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
        selected_features = st.multiselect(
            "Select Features for Clustering (Choose 2)",
            options=numerical_features,
            default=["Tumor Size", "Age"]
        )

        if len(selected_features) == 2:
            X_clustering = data[selected_features]

            # Elbow Method
            if st.checkbox("Show Elbow Method"):
                inertias = []
                K = range(1, 10)
                for k in K:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(X_clustering)
                    inertias.append(kmeans.inertia_)

                fig = px.line(x=K, y=inertias,
                              title='Elbow Method for Optimal k',
                              labels={'x': 'k', 'y': 'Inertia'})
                st.plotly_chart(fig)

            # K-Means Clustering
            n_clusters = st.slider("Select Number of Clusters", 2, 8, 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            data['Cluster'] = kmeans.fit_predict(X_clustering)

            fig = px.scatter(data, x=selected_features[0], y=selected_features[1],
                             color='Cluster',
                             title='K-Means Clustering Results')
            st.plotly_chart(fig)

            # Silhouette Score
            silhouette_avg = silhouette_score(X_clustering, data['Cluster'])
            st.write(f"Silhouette Score: {silhouette_avg:.3f}")

    else:  # Regression
        st.write("### Survival Months Prediction")

        # Feature and Target Selection
        X = data[['Age', 'Tumor Size', 'Reginol Node Positive']]
        y = data['Survival Months']

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model Training
        reg_models = {
            'Linear': LinearRegression(),
            'Random Forest': RandomForestRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42)
        }

        reg_results = {}
        for name, model in reg_models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            reg_results[name] = {
                'R2': r2_score(y_test, y_pred),
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred)
            }

        # Results Visualization
        reg_results_df = pd.DataFrame(reg_results).T
        fig = px.bar(reg_results_df, barmode='group',
                     title='Regression Model Performance')
        st.plotly_chart(fig)

        # Learning Curves
        if st.checkbox("Show Learning Curves"):
            selected_model = st.selectbox("Select Model", list(reg_models.keys()))
            train_sizes, train_scores, test_scores = learning_curve(
                reg_models[selected_model], X, y, cv=5,
                train_sizes=np.linspace(0.1, 1.0, 10))

            fig = px.line(x=train_sizes,
                          y=[train_scores.mean(axis=1), test_scores.mean(axis=1)],
                          title=f'Learning Curves - {selected_model}',
                          labels={'x': 'Training Examples', 'y': 'Score'})
            st.plotly_chart(fig)

# Tab 8: Advanced Data Cleaning and Preprocessing
with tab8:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("Advanced Data Cleaning and Preprocessing")

    st.write("### Missing Value Analysis")
    st.write("Below is the missing data summary for the dataset:")
    missing_data = data.isnull().sum()
    missing_percentage = (missing_data / len(data)) * 100
    missing_summary = pd.DataFrame({
        "Missing Values": missing_data,
        "Percentage": missing_percentage
    }).sort_values(by="Percentage", ascending=False)
    st.write(missing_summary)

    st.write("### Imputation Options")
    imputation_method = st.radio(
        "Select an imputation method for missing values:",
        ["Mean Imputation", "KNN Imputation", "Drop Rows"]
    )

    if imputation_method == "Mean Imputation":
        mean_imputer = SimpleImputer(strategy='mean')
        data_imputed = pd.DataFrame(mean_imputer.fit_transform(data[numeric_filter]), columns=numeric_filter)
        st.write("Missing values have been filled using the mean of each column.")
    elif imputation_method == "KNN Imputation":
        knn_imputer = KNNImputer(n_neighbors=5)
        data_imputed = pd.DataFrame(knn_imputer.fit_transform(data[numeric_filter]), columns=numeric_filter)
        st.write("Missing values have been filled using KNN Imputation.")
    elif imputation_method == "Drop Rows":
        data_imputed = data.dropna()
        st.write("Rows with missing values have been dropped.")

    st.write("### Cleaned Data Preview")
    st.write(data_imputed.head())

    # Encoding Categorical Variables
    encoding_method = st.selectbox("Choose encoding method", ("Label Encoding", "One-Hot Encoding"))
    if encoding_method == "Label Encoding":
        label_column = st.selectbox("Select column for Label Encoding", data.select_dtypes(include=['object']).columns)
        label_encoder = LabelEncoder()
        data_encoded = data.copy()
        data_encoded[label_column] = label_encoder.fit_transform(data[label_column])
        st.write(f"Label Encoded Data for {label_column}:", data_encoded.head())
    elif encoding_method == "One-Hot Encoding":
        data = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns)
        st.write("One-Hot Encoded Data:", data.head())

    # Normalization and Scaling
    scale_method = st.selectbox("Choose scaling method", ("Min-Max Scaling", "Standardization", "Robust Scaling"))
    if scale_method == "Min-Max Scaling":
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
        data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
        st.write("Min-Max Scaled Data:", data.head())
    elif scale_method == "Standardization":
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
        data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
        st.write("Standardized Data:", data.head())
    elif scale_method == "Robust Scaling":
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
        data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
        st.write("Robust Scaled Data:", data.head())

    # Feature Engineering: Extracting Date-Time Features
    if "date" in data.columns:
        data['year'] = pd.to_datetime(data['date']).dt.year
        data['month'] = pd.to_datetime(data['date']).dt.month
        data['weekday'] = pd.to_datetime(data['date']).dt.weekday
        st.write("Extracted Date Features:", data.head())

    # Binning continuous features (e.g., age)
    if "age" in data.columns:
        data['age_group'] = pd.cut(data['age'], bins=[0, 18, 35, 50, 100], labels=['0-18', '19-35', '36-50', '51+'])
        st.write("Binned Age Groups:", data.head())

    # Handling Outliers: Z-Score and IQR Methods
    outlier_method = st.selectbox("Choose Outlier Detection Method", ("Z-Score Method", "IQR Method"))

    if outlier_method == "Z-Score Method":
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

        if len(numeric_columns) == 0:
            st.error("No numeric columns available for Z-score calculation.")
        else:
            # Handle missing values
            if data.isnull().sum().any():
                st.warning("Data contains missing values. Proceeding to handle them.")
                data = data.dropna()  # You can choose to fill NaN values if needed

            # Z-Score Outlier Removal
            z_scores = stats.zscore(data[numeric_columns])
            abs_z_scores = np.abs(z_scores)
            data_cleaned = data[(abs_z_scores < 3).all(axis=1)]  # Removing rows with z-score > 3
            st.write("Data after Z-Score Outlier Removal:", data_cleaned.head())

    elif outlier_method == "IQR Method":
        numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
        Q1 = data[numeric_columns].quantile(0.25)
        Q3 = data[numeric_columns].quantile(0.75)
        IQR = Q3 - Q1
        filtered_data = data[numeric_columns][
            ~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
        st.write("Data after IQR Outlier Removal:", filtered_data.head())

    # Handling Imbalanced Data: SMOTE
    imbalanced = st.checkbox("Apply SMOTE to Handle Imbalanced Data")
    if imbalanced:
        num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)

        # Add cluster labels to the dataset
        data["Cluster"] = cluster_labels
        st.write("Data with Cluster Labels:", data.head())

        # Balance clusters (if needed)
        cluster_counts = data["Cluster"].value_counts()
        st.write("Cluster Counts Before Balancing:", cluster_counts)

        # Resampling logic: Duplicate rows from smaller clusters
        max_cluster_size = cluster_counts.max()
        balanced_data = pd.concat(
            [data[data["Cluster"] == cluster].sample(max_cluster_size, replace=True, random_state=42)
             for cluster in data["Cluster"].unique()],
            axis=0
        )

        st.write("Balanced Data After Resampling:")
        st.write(balanced_data)

    st.write("### Complex Data Integration Example")
    st.markdown(
        """
        Complex data integration techniques are essential for merging datasets or enriching the dataset with external data sources.
        Here, we demonstrate:
        - Merging the dataset with a simulated external data source.
        """
    )

    # Example of merging with external simulated data
    external_data = pd.DataFrame({
        'Age': sorted(data['Age'].unique()),
        'Life Expectancy': np.random.randint(70, 85, size=len(data['Age'].unique()))
    })

    merged_data = pd.merge(data, external_data, on='Age', how='left')

    st.write("### Merged Data Preview")
    st.write(merged_data.head())

    st.write("### Next Steps")
    st.markdown(
        """
        - Perform feature engineering on the merged data.
        - Evaluate how integrated data improves predictive modeling.
        """
    )

    # Advanced Data Cleaning and EDA Tab
    with tab9:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Advanced Data Analysis & Preprocessing")

        # 1. Data Quality Analysis
        st.subheader("1. Data Quality Overview")

        # Missing value analysis
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100

        quality_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Missing Percentage': missing_percent,
            'Data Type': data.dtypes
        })

        st.write(quality_df)

        # 2. Statistical Analysis
        st.subheader("2. Statistical Analysis")

        # Numeric columns analysis
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

        if st.checkbox("Show Detailed Statistical Analysis"):
            stats_df = data[numeric_cols].agg([
                'mean', 'median', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
                'skew', 'kurtosis'
            ]).round(2)

            stats_df.index = ['Mean', 'Median', 'Std Dev', 'Min', 'Max',
                              '25th Percentile', '75th Percentile',
                              'Skewness', 'Kurtosis']
            st.write(stats_df)

        # 3. Advanced Visualizations
        st.subheader("3. Advanced Visualizations")

        # Visualization 1: Distribution Analysis
        if st.checkbox("Show Distribution Analysis"):
            selected_num_col = st.selectbox("Select Column for Distribution", numeric_cols)

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=('Distribution Plot', 'Box Plot'))

            # Add histogram
            fig.add_trace(
                go.Histogram(x=data[selected_num_col], name="Distribution"),
                row=1, col=1
            )

            # Add box plot
            fig.add_trace(
                go.Box(x=data[selected_num_col], name="Box Plot"),
                row=2, col=1
            )

            fig.update_layout(height=800, title_text=f"Distribution Analysis of {selected_num_col}")
            st.plotly_chart(fig)

        # Visualization 2: Time Series Analysis
        if st.checkbox("Show Survival Analysis"):
            fig = px.line(data.groupby('Age')['Survival Months'].mean().reset_index(),
                          x='Age', y='Survival Months',
                          title='Average Survival Months by Age')
            st.plotly_chart(fig)

        # Visualization 3: Feature Relationships
        if st.checkbox("Show Feature Relationships"):
            selected_features = st.multiselect("Select Features for Analysis",
                                               numeric_cols,
                                               default=numeric_cols[:3])

            if len(selected_features) > 0:
                correlation_matrix = data[selected_features].corr()

                fig = px.imshow(correlation_matrix,
                                labels=dict(color="Correlation"),
                                x=correlation_matrix.columns,
                                y=correlation_matrix.columns,
                                color_continuous_scale='RdBu')

                st.plotly_chart(fig)

        # Visualization 4: Categorical Analysis
        if st.checkbox("Show Categorical Analysis"):
            categorical_cols = data.select_dtypes(include=['object']).columns
            selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)

            value_counts = data[selected_cat].value_counts().reset_index()
            value_counts.columns = ['Category', 'Count']

            fig = px.pie(value_counts,
                         values='Count',
                         names='Category',
                         title=f'Distribution of {selected_cat}')
            st.plotly_chart(fig)

        # Visualization 5: Bivariate Analysis
        if st.checkbox("Show Bivariate Analysis"):
            num_col = st.selectbox("Select Numeric Feature", numeric_cols, key='bivar_num')
            cat_col = st.selectbox("Select Categorical Feature", categorical_cols, key='bivar_cat')

            fig = px.violin(data, x=cat_col, y=num_col,
                            box=True, points="all",
                            title=f'Distribution of {num_col} across {cat_col}')
            st.plotly_chart(fig)

        # 4. Outlier Detection
        st.subheader("4. Outlier Detection")

        if st.checkbox("Show Outlier Analysis"):
            selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)

            Q1 = data[selected_col].quantile(0.25)
            Q3 = data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[selected_col] < (Q1 - 1.5 * IQR)) |
                            (data[selected_col] > (Q3 + 1.5 * IQR))]

            st.write(f"Number of outliers detected: {len(outliers)}")

            fig = px.box(data, y=selected_col,
                         title=f'Outlier Analysis for {selected_col}')
            st.plotly_chart(fig)
# Feature Engineering Tab
with tab10:
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.header("Data Processing & Feature Engineering")

    # 1. Feature Creation Section
    st.subheader("1. Feature Creation")

    # Age Groups
    if st.checkbox("Create Age Groups"):
        age_mean = data_actual['Age'].mean()
        age_std = data_actual['Age'].std()

        actual_age = (data_actual['Age'] * age_std) + age_mean

        data['Age_Group'] = pd.cut(data_actual['Age'],
                                   bins=[0, 30, 45, 60, 75, 100],
                                   labels=['Young', 'Middle', 'Senior', 'Elder', 'Advanced'])

        # Visualize age distribution with calculated statistics
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Age Distribution', 'Age Groups'))

        # Original age distribution
        fig.add_trace(
            go.Histogram(x=data_actual['Age'], name="Age Distribution"),
            row=1, col=1
        )

        # Age groups distribution
        age_group_counts = data['Age_Group'].value_counts()
        fig.add_trace(
            go.Bar(x=age_group_counts.index, y=age_group_counts.values, name="Age Groups"),
            row=1, col=2
        )

        fig.update_layout(height=400, title_text="Age Analysis")
        st.plotly_chart(fig)

    # Survival Risk Score
    if st.checkbox("Generate Survival Risk Score"):
        data['Risk_Score'] = (
                data['Tumor Size'] * 0.3 +
                data['Reginol Node Positive'] * 0.4 +
                data['Age'] * 0.3
        ).round(2)

        fig = px.histogram(data, x='Risk_Score',
                           title='Distribution of Risk Scores',
                           nbins=30)
        st.plotly_chart(fig)

    # 2. Feature Transformation
    st.subheader("2. Advanced Transformations")

    transform_type = st.selectbox(
        "Select Transformation Method",
        ["Log Transform", "Box-Cox", "Yeo-Johnson", "Quantile"]
    )

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("Select Column for Transformation", numeric_cols)

    if transform_type and selected_col:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=('Original Distribution', 'Transformed Distribution'))

        # Original Distribution
        fig.add_trace(
            go.Histogram(x=data[selected_col], name="Original"),
            row=1, col=1
        )

        # Transform data based on selection
        if transform_type == "Log Transform":
            transformed_data = np.log1p(data[selected_col])
        elif transform_type == "Box-Cox":
            transformed_data = stats.boxcox(data[selected_col] + 1)[0]
        elif transform_type == "Yeo-Johnson":
            transformed_data = stats.yeojohnson(data[selected_col])[0]
        else:  # Quantile
            transformer = QuantileTransformer(output_distribution='normal')
            transformed_data = transformer.fit_transform(data[selected_col].values.reshape(-1, 1)).flatten()

        # Transformed Distribution
        fig.add_trace(
            go.Histogram(x=transformed_data, name="Transformed"),
            row=1, col=2
        )

        fig.update_layout(height=400, title_text=f"{transform_type} Transformation")
        st.plotly_chart(fig)

    # 3. Feature Interactions
    st.subheader("3. Feature Interactions")

    if st.checkbox("Generate Interaction Features"):
        selected_features = st.multiselect(
            "Select 2 Features for Interaction",
            numeric_cols,
            default=numeric_cols[:2]
        )

        if len(selected_features) >= 2:
            # Multiplication Interaction
            data[f'{selected_features[0]}_{selected_features[1]}_interaction'] = (
                    data[selected_features[0]] * data[selected_features[1]]
            )

            # Ratio Interaction
            data[f'{selected_features[0]}_{selected_features[1]}_ratio'] = (
                    data[selected_features[0]] / (data[selected_features[1]] + 1)
            )

            st.write("New Interaction Features:")
            st.write(data[[f'{selected_features[0]}_{selected_features[1]}_interaction',
                           f'{selected_features[0]}_{selected_features[1]}_ratio']].describe())

    # 4. Dimensionality Reduction
    st.subheader("4. Dimensionality Reduction")

    dim_reduction = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "t-SNE", "UMAP"]
    )

    if st.checkbox("Apply Dimensionality Reduction"):
        # Prepare numeric data
        X = data[numeric_cols].fillna(0)

        if dim_reduction == "PCA":
            reducer = PCA(n_components=2)
            reduced_data = reducer.fit_transform(StandardScaler().fit_transform(X))
        elif dim_reduction == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(StandardScaler().fit_transform(X))
        elif dim_reduction == "UMAP":
            reducer = UMAP(n_components=2, random_state=42)
            reduced_data = reducer.fit_transform(StandardScaler().fit_transform(X))

        # Visualization
        fig = px.scatter(
            x=reduced_data[:, 0], y=reduced_data[:, 1],
            color=data['Status'],
            title=f'{dim_reduction} Visualization'
        )
        st.plotly_chart(fig)

