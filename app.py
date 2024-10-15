# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the app
st.title("Breast Cancer Detection Analysis")

# Upload the dataset
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the data
    bc_data = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("### Dataset", bc_data)

    # Data Cleaning
    if st.sidebar.button("Clean Data"):
        # Remove duplicates
        bc_data.drop_duplicates(inplace=True)

        # Fill missing values
        numeric_cols_bc = bc_data.select_dtypes(include=['float64', 'int64']).columns
        bc_data[numeric_cols_bc] = bc_data[numeric_cols_bc].fillna(bc_data[numeric_cols_bc].mean())
        categorical_cols_bc = bc_data.select_dtypes(include=['object']).columns
        for column in categorical_cols_bc:
            bc_data[column].fillna(bc_data[column].mode()[0], inplace=True)
        st.write("Data has been cleaned.")

    # Statistical Summary
    if st.sidebar.checkbox("Show Statistical Summary"):
        st.write("### Statistical Summary of Numerical Features")
        st.write(bc_data.describe())

    # Visualization: T Stage Distribution
    if st.sidebar.checkbox("Show T Stage Distribution"):
        fig = px.bar(bc_data, x='T Stage ', title="Distribution of T Stage",
                     labels={'T Stage ': 'T Stage Level', 'count': 'Number of Patients'})
        st.plotly_chart(fig)

    # Visualization: Tumor Size vs Lymph Nodes
    if st.sidebar.checkbox("Show Tumor Size vs Lymph Nodes"):
        if 'Lymph Nodes' in bc_data.columns:
            fig = px.scatter(bc_data, x='Tumor Size', y='Lymph Nodes',
                             color='T Stage ', title="Tumor Size vs. Lymph Nodes",
                             labels={'Tumor Size': 'Tumor Size (mm)', 'Lymph Nodes': 'Lymph Nodes Affected'})
            st.plotly_chart(fig)
        else:
            st.warning("Lymph Nodes column not found in the dataset.")

    # Visualization: Correlation Matrix
    if st.sidebar.checkbox("Show Correlation Matrix"):
        corr = bc_data.corr()
        fig = px.imshow(corr, title="Correlation Matrix")
        st.plotly_chart(fig)

    # Additional features, like user input, predictions, etc., can be added here

else:
    st.warning("Please upload a CSV file.")
