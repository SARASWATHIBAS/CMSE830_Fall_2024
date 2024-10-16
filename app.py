import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np

# Function to load the data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Function for encoding categorical variables
def encode_data(data):
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = label_encoder.fit_transform(data[col])
    return data

# Streamlit App
st.title("Breast Cancer Analysis App")
uploaded_file = st.file_uploader("Upload your Breast Cancer Dataset (CSV)", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)
    st.success("Data loaded successfully!")

    # Sidebar filters
    st.sidebar.header("Filter Data")
    categorical_filter = st.sidebar.multiselect("Select Categorical Columns to View",
                                                data.select_dtypes(include='object').columns)
    numeric_filter = st.sidebar.multiselect("Select Numeric Columns to View",
                                            data.select_dtypes(include=np.number).columns)

    # Tabs for different sections
    tabs = st.tabs(["Data Overview", "Visualization", "Imputation Comparison", "Encoding and Scaling"])

    with tabs[0]:
        st.write("### Data Overview")
        if categorical_filter:
            st.write("#### Categorical Data")
            st.write(data[categorical_filter].head())

        if numeric_filter:
            st.write("#### Numerical Data")
            st.write(data[numeric_filter].describe())
        else:
            st.write("No numeric columns selected.")

    with tabs[1]:
        st.write("### Visualizations")
        if st.checkbox("Show Correlation Heatmap"):
            st.subheader("Correlation Heatmap")
            if numeric_filter:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(data[numeric_filter])
                corr_matrix = np.corrcoef(scaled_data.T)

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, xticklabels=numeric_filter, yticklabels=numeric_filter, cmap='coolwarm')
                st.pyplot(fig)
            else:
                st.warning("Please select numeric columns for correlation analysis.")

        # Scatter Plot for Numerical vs Categorical
        st.subheader("Scatter Plot")
        x_axis = st.selectbox("Select X-axis", numeric_filter)
        y_axis = st.selectbox("Select Y-axis", numeric_filter)
        hue = st.selectbox("Select Hue (Categorical)", categorical_filter)

        if x_axis and y_axis and hue:
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], hue=data[hue], palette='husl', ax=ax)
            ax.set_title(f'Scatter Plot: {y_axis} vs {x_axis} colored by {hue}')
            st.pyplot(fig)

    with tabs[2]:
        st.write("### Compare Imputation Methods")
        # Data Imputation Comparison
        if st.checkbox("Run Imputation Comparison"):
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

    with tabs[3]:
        st.write("### Encoding and Scaling")
        if st.button("Encode Data"):
            encoded_data = encode_data(data)
            st.write("Encoded Data")
            st.write(encoded_data.head())

        if st.button("Scale Data with Min-Max Scaling"):
            scaler = MinMaxScaler()
            scaled_data = pd.DataFrame(scaler.fit_transform(data[numeric_filter]), columns=numeric_filter)
            st.write("Scaled Data")
            st.write(scaled_data.head())

else:
    st.info("Please upload a CSV file to get started.")
