#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load breast cancer dataset and an additional dataset on breast cancer
def load_data():
    try:
        bc_data = pd.read_csv('SEER Breast Cancer Dataset .csv')
        return bc_data
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None

bc_data = load_data()

# Display the first few rows of both datasets to understand their structure
(bc_data.head())
# Display the dataset overview and first few rows
print("Dataset Overview:")
print(bc_data.info())


# Remove duplicates from the breast cancer dataset
bc_data.drop_duplicates(inplace=True)

# Check for missing values in the breast cancer dataset
print("Missing values in Breast Cancer dataset:")
print(bc_data.isnull().sum())

# Fill missing values in numerical columns with the mean
numeric_cols_bc = bc_data.select_dtypes(include=['float64', 'int64']).columns
bc_data[numeric_cols_bc] = bc_data[numeric_cols_bc].fillna(bc_data[numeric_cols_bc].mean())

# For categorical columns, fill missing values with the mode
categorical_cols_bc = bc_data.select_dtypes(include=['object']).columns
for column in categorical_cols_bc:
    bc_data[column].fillna(bc_data[column].mode()[0], inplace=True)

# Optionally, display the cleaned dataset to verify changes
print("\nCleaned Breast Cancer dataset:")
print(bc_data.isnull().sum())


# In[12]:


from sklearn.preprocessing import LabelEncoder

# 1. Check the data types of each column
print("Data types in the Breast Cancer dataset:")
print(bc_data.dtypes)

# 3. Encoding categorical variables

# List of ordinal columns to encode
ordinal_cols = ['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage']

# Identify remaining categorical columns (excluding the ordinal ones)
categorical_cols = bc_data.select_dtypes(include=['object']).columns
categorical_cols = [col for col in categorical_cols if col not in ordinal_cols]


# Apply Label Encoding to each ordinal column
label_encoder = LabelEncoder()

for col in ordinal_cols:
    if col in bc_data.columns:  # Ensure column exists in the dataset
        bc_data[col] = label_encoder.fit_transform(bc_data[col])

# Display the updated dataset after encoding
print("\nSEER Dataset After Label Encoding of Ordinal Columns:")
print(bc_data[ordinal_cols].head())


# Option 2: One-Hot Encoding for nominal categorical variables
# This will create new binary columns for each category in categorical variables
bc_data = pd.get_dummies(bc_data, columns=categorical_cols, drop_first=True)

# Display the first few rows of the modified dataset to verify encoding
print("\nModified Breast Cancer dataset after encoding:")
print(bc_data.head())

# Check the data types again after encoding
print("\nData types after encoding:")
print(bc_data.dtypes)


# Drop the Unnamed: 3 column (or any irrelevant empty column)
bc_data = bc_data.loc[:, ~bc_data.columns.str.contains('^Unnamed')]

# Verify the columns after dropping
print("Columns after removing 'Unnamed' columns:")
print(bc_data.columns)


# In[52]:


# 1. Basic statistical summary of numerical columns
print("Statistical Summary of Numerical Features:")
print(bc_data.describe())

# 2. Statistical summary of categorical/encoded columns
print("\nStatistical Summary of Encoded Features:")
print(bc_data.describe(include='all'))

# 3. Count of unique values in ordinal features
for col in ['T Stage ', 'N Stage', '6th Stage', 'Grade', 'A Stage']:
    print(f"\nUnique values count for {col}:")
    print(bc_data[col].value_counts())


# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


# Ensure the 'Age' column exists and is numerical
if 'Age' in bc_data.columns:
    fig = px.box(bc_data, x='Grade', y='Age',
                 title="Distribution of Age Across Different Grades",
                 labels={'Grade': 'Tumor Grade', 'Age': 'Patient Age'})
    fig.show()


# In[32]:


import plotly.express as px

# Count plot with color grouping for T Stage
fig = px.histogram(
    bc_data, 
    x='T Stage ', 
    color='T Stage ',  # Group bars by T Stage levels
    title="Distribution of T Stage Levels",
    labels={'T Stage ': 'T Stage Level', 'count': 'Number of Patients'},
    text_auto=True,  # Display counts on bars
    color_discrete_sequence=px.colors.qualitative.Plotly  # Vibrant color palette
)

# Customize layout for better readability
fig.update_layout(
    title_font=dict(size=22),
    xaxis_title='T Stage Level',
    yaxis_title='Number of Patients',
    xaxis=dict(tickfont=dict(size=14)),  # Larger axis labels
    yaxis=dict(tickfont=dict(size=14)),
    bargap=0.2,  # Space between bars
)

fig.show()


# In[62]:


import plotly.express as px

# First, let's create a new DataFrame to aggregate the data
# Aggregate data by T Stage and Grade
agg_data = bc_data.groupby(['T Stage ', 'Grade'], as_index=False).agg(
    count=('Tumor Size', 'count'),
    avg_tumor_size=('Tumor Size', 'mean')
)

# Bar plot with color encoding for Grade and size based on average Tumor Size
fig = px.bar(
    agg_data,
    x='T Stage ',
    y='count',
    color='Grade',
    title="Patient Count by T Stage and Grade",
    labels={'T Stage ': 'Tumor Stage', 'count': 'Number of Patients'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.show()


# In[64]:


# Scatter plot showing Tumor Size vs. N Stage, with color by T Stage and size by Tumor Size
fig = px.scatter(
    bc_data,
    x='Tumor Size', 
    y='N Stage', 
    color='T Stage ',  # Color encoding by T Stage
    size='Tumor Size',  # Size encoding by Tumor Size
    title="Tumor Size vs. N Stage Colored by T Stage",
    labels={'Tumor Size': 'Tumor Size (mm)', 'N Stage': 'Lymph Node Stage'},
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.show()


# In[66]:


# Faceted box plot of Tumor Size by T Stage, separated by Grade
fig = px.box(
    bc_data,
    x='T Stage ',
    y='Tumor Size',
    color='Grade',  # Color coding by Grade
    title="Box Plot of Tumor Size by T Stage and Grade",
    labels={'T Stage ': 'Tumor Stage', 'Tumor Size': 'Tumor Size (mm)'},
    facet_col='Grade',  # Faceting by Grade
    color_discrete_sequence=px.colors.qualitative.Plotly
)

fig.show()


# In[70]:


import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)

# Introduce missingness randomly in 10% of the 'Age' column (numerical)
num_missing = int(0.1 * bc_data.shape[0])  # 10% of the rows
random_indices = np.random.choice(bc_data.index, num_missing, replace=False)
bc_data.loc[random_indices, 'Age'] = np.nan

# Introduce missingness in the 'T Stage' column (categorical)
random_indices = np.random.choice(bc_data.index, num_missing, replace=False)
bc_data.loc[random_indices, 'T Stage '] = np.nan

# Check for missing values
print("Missing values after introducing random missingness:")
print(bc_data.isnull().sum())


# Impute missing values in 'Age' with the mean
bc_data['Age'].fillna(bc_data['Age'].mean(), inplace=True)
# Impute missing values in 'T Stage' with the mode
bc_data['T Stage '].fillna(bc_data['T Stage '].mode()[0], inplace=True)


# Check for any remaining missing values
print("\nMissing values after imputation:")
print(bc_data.isnull().sum())


# In[82]:


import plotly.express as px

# Create the bar plot for T Stage distribution after imputation
fig = px.bar(
    bc_data,
    x='T Stage ',  # Ensure the column name is correct
    title="Distribution of T Stage (After Imputation)",
    labels={'T Stage ': 'T Stage Level', 'count': 'Number of Patients'},
    color='T Stage ',  # Color bars based on T Stage
    color_discrete_sequence=px.colors.qualitative.Plotly  # Set a color sequence
)

# Update layout for better visibility
fig.update_layout(
    title_font_size=24,  # Title font size
    xaxis_title_font_size=18,  # X-axis title font size
    yaxis_title_font_size=18,  # Y-axis title font size
    xaxis_tickfont_size=14,  # X-axis tick font size
    yaxis_tickfont_size=14,  # Y-axis tick font size
    template='plotly_white',  # Use a clean template
    xaxis=dict(showgrid=True, gridcolor='LightGrey'),  # Show gridlines
    yaxis=dict(showgrid=True, gridcolor='LightGrey')   # Show gridlines
)

# Show the improved plot
fig.show()


# In[76]:


import plotly.express as px

# Visualize the distribution of 'Age' after imputation
fig = px.histogram(bc_data, x='Age', nbins=30, title="Distribution of Age (After Imputation)",
                   labels={'Age': 'Patient Age'})
fig.show()


# In[86]:


import streamlit as st
import pandas as pd
import plotly.express as px

# Title of the app
st.title("Breast Cancer Dataset Analysis")

# Documentation
st.markdown("""
This application allows users to explore the SEER Breast Cancer dataset.
You can visualize the distribution of Tumor Stage and analyze relationships 
between different features.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Dropdown for selecting T Stage
t_stage_options = bc_data['T Stage '].unique()
selected_t_stage = st.sidebar.selectbox("Select T Stage:", options=t_stage_options)

# Dropdown for selecting N Stage
n_stage_options = bc_data['N Stage'].unique()
selected_n_stage = st.sidebar.selectbox("Select N Stage:", options=n_stage_options)

# Filter data based on selections
filtered_data = bc_data[(bc_data['T Stage '] == selected_t_stage) & (bc_data['N Stage'] == selected_n_stage)]

# Bar plot for T Stage distribution
fig_t_stage = px.bar(filtered_data, 
                      x='T Stage ', 
                      title="Distribution of T Stage for Selected N Stage",
                      labels={'T Stage ': 'T Stage Level'},
                      color='T Stage ',
                      color_discrete_sequence=px.colors.qualitative.Plotly)

st.plotly_chart(fig_t_stage)

# Scatter plot for Tumor Size vs Lymph Nodes (if applicable)
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






