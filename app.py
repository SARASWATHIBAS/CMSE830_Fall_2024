import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
import numpy as np

# Load the dataset
st.title("Breast Cancer Analysis App")
uploaded_file = st.file_uploader("Upload your Breast Cancer Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Remove any unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    # Sidebar filters
    st.sidebar.header("Filter Data")
    categorical_filter = st.sidebar.multiselect("Select Categorical Columns to View",
                                                data.select_dtypes(include='object').columns)
    numeric_filter = st.sidebar.multiselect("Select Numeric Columns to View",
                                            data.select_dtypes(include=np.number).columns)

    # Data Overview
    if st.checkbox("Show Data Overview"):
        st.subheader("Data Overview")
        st.write(data.describe(include='all'))

    # Correlation Heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_filter])
        corr_matrix = np.corrcoef(scaled_data.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=numeric_filter, yticklabels=numeric_filter, cmap='coolwarm')
        st.pyplot(fig)

    # Data Imputation Comparison
    if st.checkbox("Compare Imputation Methods"):
        st.subheader("Imputation Methods: Mean vs KNN")

        # Mean Imputation
        mean_imputer = SimpleImputer(strategy='mean')
        data_mean_imputed = pd.DataFrame(mean_imputer.fit_transform(data[numeric_filter]),
                                         columns=numeric_filter)

        # KNN Imputation
        knn_imputer = KNNImputer(n_neighbors=5)
        data_knn_imputed = pd.DataFrame(knn_imputer.fit_transform(data[numeric_filter]),
                                        columns=numeric_filter)

        # Show distribution comparisons
        for col in numeric_filter:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            sns.histplot(data_mean_imputed[col], kde=True, ax=ax[0], color='skyblue')
            ax[0].set_title(f'Mean Imputed: {col}')

            sns.histplot(data_knn_imputed[col], kde=True, ax=ax[1], color='salmon')
            ax[1].set_title(f'KNN Imputed: {col}')

            st.pyplot(fig)

    # Min-Max Scaling
    if st.checkbox("Show Min-Max Scaling"):
        st.subheader("Min-Max Scaling")
        min_max_scaler = MinMaxScaler()
        scaled_data_min_max = pd.DataFrame(min_max_scaler.fit_transform(data[numeric_filter]),
                                            columns=numeric_filter)

        st.write(scaled_data_min_max.head())

    # Dynamic Relationship Inference
    st.subheader("Dynamic Relationship Inference")

    # Select features for analysis
    feature_x = st.selectbox("Select Feature X", numeric_filter)
    feature_y = st.selectbox("Select Feature Y", numeric_filter)

    if feature_x and feature_y:
        # Scatter plot for the selected features
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=feature_x, y=feature_y)
        ax.set_title(f'Scatter Plot: {feature_x} vs {feature_y}')
        st.pyplot(fig)

        # Calculate and display correlation
        correlation = data[feature_x].corr(data[feature_y])
        st.write(f"### Correlation Coefficient between {feature_x} and {feature_y}: {correlation:.2f}")

        # Interpretation
        if correlation > 0.5:
            st.write("### Interpretation: Strong positive relationship.")
        elif correlation < -0.5:
            st.write("### Interpretation: Strong negative relationship.")
        else:
            st.write("### Interpretation: Weak or no relationship.")

    # Scatter Plot for Numerical vs Categorical
    st.subheader("Scatter Plot")
    x_axis = st.selectbox("Select X-axis", numeric_filter)
    y_axis = st.selectbox("Select Y-axis", numeric_filter)
    hue = st.selectbox("Select Hue (Categorical)", categorical_filter)

    if x_axis and y_axis and hue:
        fig, ax = plt.subplots()
        sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data[hue], palette='husl', ax=ax)
        st.pyplot(fig)

    # Closing message
    st.write("### Thank you for using the Breast Cancer Analysis App!")
