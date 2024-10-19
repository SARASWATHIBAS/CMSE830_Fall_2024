import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import plotly.express as px

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/main/SEER%20Breast%20Cancer%20Dataset%20.csv"


# Add background image
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/blob/main/back_drop.png');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Breast Cancer Analysis App")

try:
    data = pd.read_csv(url)
except Exception as e:
    st.error(f"Error loading data: {e}")

# Remove any unnamed columns
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Sidebar filters
st.sidebar.header("Filter Data")

# Multi-select for categorical and numeric features
categorical_filter = data.select_dtypes(include='object').columns.tolist()
numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

selected_categorical = st.sidebar.multiselect("Select Categorical Columns", categorical_filter)
selected_numeric = st.sidebar.multiselect("Select Numeric Columns", numeric_filter)

# Tabs for app sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Data Overview", "Correlation Heatmap", "Imputation Comparison", "Scaling", "Visualizations", "Search"]
)

# Data Overview Tab
with tab1:
    st.subheader("Data Overview")
    # Static Overview
    st.header("Data Overview")
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
    - **Regional Node Examined/Positive:** Number of nodes examined and found positive.
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

# Correlation Heatmap Tab
with tab2:
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
with tab3:
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
with tab4:
    st.subheader("Min-Max Scaling")
    if selected_numeric:
        min_max_scaler = MinMaxScaler()
        scaled_data_min_max = pd.DataFrame(min_max_scaler.fit_transform(data[selected_numeric]),
                                           columns=selected_numeric)

        st.write(scaled_data_min_max.head())
    else:
        st.write("Please select numeric columns for min-max scaling.")

# Advanced Visualizations Tab
with tab5:
    st.subheader("Advanced Visualizations")

    # Select features for dynamic plots
    feature_x = st.selectbox("Select Feature X", selected_numeric)
    feature_y = st.selectbox("Select Feature Y", selected_numeric)
    hue_feature = st.selectbox("Select Hue (Categorical)", selected_categorical)

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

# Search Tab with Dropdown for Categorical Variables
with tab6:
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