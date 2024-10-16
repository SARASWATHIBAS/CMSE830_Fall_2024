import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import plotly.express as px

# Load the dataset
st.title("Breast Cancer Analysis App")
uploaded_file = st.file_uploader("Upload your Breast Cancer Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Remove 'Unnamed: 3' if present
    if 'Unnamed: 3' in data.columns:
        data.drop(columns=['Unnamed: 3'], inplace=True)

    # Data Overview
    st.subheader("Data Overview")
    st.write(data.head())

    st.write("""
    ### Description of Data
    - **Age**: Patient’s age group.
    - **Race**: Racial classification.
    - **Marital Status**: Current marital status of the patient.
    - **T Stage**: Tumor stage based on size.
    - **N Stage**: Lymph node involvement stage.
    - **6th Stage**: Stage classification according to the 6th edition of AJCC.
    - **Grade**: Tumor grade indicating aggressiveness.
    - **A Stage**: Distant metastasis presence/absence.
    - **Tumor Size**: Size of the tumor in mm.
    - **Estrogen/Progesterone Status**: Hormone receptor status.
    - **Regional Node Examined/Positive**: Number of nodes examined and found positive.
    - **Survival Months**: Months patient survived.
    - **Status**: Patient’s survival status.
    """)

    # Sidebar for filters
    st.sidebar.header("Filter Data")
    categorical_filter = st.sidebar.multiselect("Select Categorical Columns to View",
                                                data.select_dtypes(include='object').columns)
    numeric_filter = st.sidebar.multiselect("Select Numeric Columns to View",
                                            data.select_dtypes(include=np.number).columns)

    # Display filtered data
    if categorical_filter:
        st.write("### Categorical Data Overview")
        st.write(data[categorical_filter].head())

    if numeric_filter:
        st.write("### Numerical Data Overview")
        st.write(data[numeric_filter].describe())

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_filter])
        corr_matrix = np.corrcoef(scaled_data.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=numeric_filter, yticklabels=numeric_filter, cmap='coolwarm')
        st.pyplot(fig)

        st.write("### Interpretation:")
        st.write(
            "The correlation heatmap visualizes the relationships between numerical features. Darker colors indicate stronger correlations, either positive or negative. Features with high correlations may provide insights into how they interact with each other in the context of breast cancer.")

    # Data Imputation Comparison
    if st.checkbox("Compare Imputation Methods"):
        st.subheader("Imputation Methods: Mean vs KNN")

        # Mean Imputation
        mean_imputer = SimpleImputer(strategy='mean')
        data_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(data[numeric_filter]), columns=numeric_filter)

        # KNN Imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data[numeric_filter]), columns=numeric_filter)

        # Show distribution comparisons
        for col in numeric_filter:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(data_mean_imputed[col], kde=True, ax=ax[0], color='skyblue')
            ax[0].set_title(f'Mean Imputed: {col}')

            sns.histplot(data_knn_imputed[col], kde=True, ax=ax[1], color='salmon')
            ax[1].set_title(f'KNN Imputed: {col}')

            st.pyplot(fig)

            st.write(f"### Interpretation for {col}:")
            st.write(
                "The histograms show the distribution of the values for the mean and KNN imputation methods. KNN may better capture the underlying data distribution by considering the nearest neighbors, whereas mean imputation might lead to underestimating variability.")

    # Scatter Plot for Numerical vs Categorical
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_filter)
    y_axis = st.selectbox("Select Y-axis", numeric_filter)
    hue = st.selectbox("Select Hue (Categorical)", categorical_filter)

    if x_axis and y_axis and hue:
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data[hue], palette='husl', ax=ax)
        st.pyplot(fig)

        st.write("### Interpretation:")
        st.write(
            "The scatter plot illustrates the relationship between two numerical features, colored by the categorical variable. Clustering or separation can indicate how the categorical variable affects the numerical features, revealing trends or patterns.")

    # Min-Max Scaling
    if st.checkbox("Apply Min-Max Scaling"):
        st.subheader("Min-Max Scaled Data")
        min_max_scaler = MinMaxScaler()
        min_max_scaled_data = pd.DataFrame(min_max_scaler.fit_transform(data[numeric_filter]), columns=numeric_filter)
        st.write(min_max_scaled_data.describe())

        st.write("### Interpretation:")
        st.write(
            "Min-Max scaling adjusts the numerical features to a range between 0 and 1. This scaling is essential for algorithms sensitive to feature scales, helping to improve model performance.")

    # Advanced Visualizations
    st.sidebar.header("Advanced Visualizations")
    if st.sidebar.checkbox("Show Pair Plot"):
        st.subheader("Pair Plot of Numerical Features")
        fig = sns.pairplot(data[numeric_filter])
        st.pyplot(fig)

        st.write("### Interpretation:")
        st.write(
            "The pair plot visualizes pairwise relationships in the dataset. It can reveal distributions and correlations between features, helping to identify trends or clusters.")

    # Interactive Visualizations with Plotly
    if st.sidebar.checkbox("Show Interactive Scatter Plot"):
        st.subheader("Interactive Scatter Plot")
        fig = px.scatter(data, x=x_axis, y=y_axis, color=hue, title="Interactive Scatter Plot")
        st.plotly_chart(fig)

        st.write("### Interpretation:")
        st.write(
            "The interactive scatter plot allows users to hover over points for more information. It provides a dynamic way to explore the relationship between selected features.")

    if st.sidebar.checkbox("Show Box Plot"):
        st.subheader("Box Plot of Numeric Features by Categorical Feature")
        categorical_col = st.selectbox("Select Categorical Column for Box Plot", categorical_filter)
        if categorical_col:
            fig = plt.figure(figsize=(12, 6))
            sns.boxplot(data=data, x=categorical_col, y=x_axis)
            st.pyplot(fig)

            st.write("### Interpretation:")
            st.write(
                "The box plot displays the distribution of the numerical feature across different categories. It highlights medians, quartiles, and potential outliers, providing insights into how the categorical variable impacts the numerical data.")

# Optional: Footer with contact information
st.sidebar.markdown("### Contact Information")
st.sidebar.markdown("For any inquiries, please contact: [Your Email](mailto:your-email@example.com)")
