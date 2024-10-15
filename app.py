# Import necessary libraries
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Function to load the breast cancer dataset
def load_data():
    try:
        bc_data = pd.read_csv('SEER Breast Cancer Dataset .csv')
        return bc_data
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        return None


# Load data
bc_data = load_data()

if bc_data is not None:
    # Display the first few rows of the dataset
    st.subheader("Dataset Overview")
    st.write(bc_data.head())

    # Data cleaning and preprocessing
    st.markdown("## Data Cleaning and Preprocessing")
    # Remove duplicates
    bc_data.drop_duplicates(inplace=True)
    # Check for missing values
    st.write("Missing values in Breast Cancer dataset:")
    st.write(bc_data.isnull().sum())

    # Fill missing values in numerical columns with the mean
    numeric_cols_bc = bc_data.select_dtypes(include=['float64', 'int64']).columns
    bc_data[numeric_cols_bc] = bc_data[numeric_cols_bc].fillna(bc_data[numeric_cols_bc].mean())

    # Fill missing values in categorical columns with the mode
    categorical_cols_bc = bc_data.select_dtypes(include=['object']).columns
    for column in categorical_cols_bc:
        bc_data[column].fillna(bc_data[column].mode()[0], inplace=True)

    # Display cleaned dataset information
    st.write("Cleaned Breast Cancer dataset:")
    st.write(bc_data.isnull().sum())

    # Encoding categorical variables
    st.markdown("## Encoding Categorical Variables")
    ordinal_cols = ['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage']
    label_encoder = LabelEncoder()

    for col in ordinal_cols:
        if col in bc_data.columns:
            bc_data[col] = label_encoder.fit_transform(bc_data[col])

    # One-hot encoding for nominal categorical variables
    categorical_cols = bc_data.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ordinal_cols]
    bc_data = pd.get_dummies(bc_data, columns=categorical_cols, drop_first=True)

    # Drop unnecessary columns
    bc_data = bc_data.loc[:, ~bc_data.columns.str.contains('^Unnamed')]

    # Display updated dataset after encoding
    st.write("Modified Breast Cancer dataset after encoding:")
    st.write(bc_data.head())

    # User Input Features
    st.sidebar.header("User Input Features")
    t_stage_options = bc_data['T Stage '].unique()
    selected_t_stage = st.sidebar.selectbox("Select T Stage:", options=t_stage_options)

    n_stage_options = bc_data['N Stage'].unique()
    selected_n_stage = st.sidebar.selectbox("Select N Stage:", options=n_stage_options)

    # Filter data based on selections
    filtered_data = bc_data[(bc_data['T Stage '] == selected_t_stage) & (bc_data['N Stage'] == selected_n_stage)]

    # Visualizations
    st.markdown("## Visualizations")

    # Bar plot for T Stage distribution
    fig_t_stage = px.bar(filtered_data,
                         x='T Stage ',
                         title="Distribution of T Stage for Selected N Stage",
                         labels={'T Stage ': 'T Stage Level'},
                         color='T Stage ',
                         color_discrete_sequence=px.colors.qualitative.Plotly)
    st.plotly_chart(fig_t_stage)

    # Scatter plot for Tumor Size vs Lymph Nodes
    if 'Tumor Size' in bc_data.columns and 'Lymph Nodes' in bc_data.columns:
        fig_scatter = px.scatter(filtered_data,
                                 x='Tumor Size',
                                 y='Lymph Nodes',
                                 color='T Stage ',
                                 title="Tumor Size vs. Lymph Nodes",
                                 labels={'Tumor Size': 'Tumor Size (mm)', 'Lymph Nodes': 'Lymph Nodes Affected'})
        st.plotly_chart(fig_scatter)

    # Basic statistics of filtered data
    st.subheader("Basic Statistics of Filtered Data")
    st.write(filtered_data.describe())

    # Box plot for Age by Grade
    if 'Age' in bc_data.columns:
        fig_age_grade = px.box(bc_data,
                               x='Grade',
                               y='Age',
                               title="Distribution of Age Across Different Grades",
                               labels={'Grade': 'Tumor Grade', 'Age': 'Patient Age'})
        st.plotly_chart(fig_age_grade)

    # Add any other visualizations as needed
else:
    st.warning("No data available to display.")
