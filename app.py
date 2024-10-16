import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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
    categorical_filter = data.select_dtypes(include='object').columns.tolist()
    numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

    # Tab structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Overview", "Correlation Heatmap", "Imputation Comparison", "Scaling", "Visualizations"])

    # Data Overview Tab
    with tab1:
        st.subheader("Data Overview")
        st.write(data.describe(include='all'))

    # Correlation Heatmap Tab
    with tab2:
        st.subheader("Correlation Heatmap")
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_filter])
        corr_matrix = np.corrcoef(scaled_data.T)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, xticklabels=numeric_filter, yticklabels=numeric_filter, cmap='coolwarm')
        st.pyplot(fig)

    # Data Imputation Comparison Tab
    with tab3:
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

    # Min-Max Scaling Tab
    with tab4:
        st.subheader("Min-Max Scaling")
        min_max_scaler = MinMaxScaler()
        scaled_data_min_max = pd.DataFrame(min_max_scaler.fit_transform(data[numeric_filter]),
                                            columns=numeric_filter)

        st.write(scaled_data_min_max.head())

    # Advanced Visualizations Tab
    with tab5:
        st.subheader("Advanced Visualizations")

        # Select features for dynamic plots
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

        # Additional visualizations
        if st.checkbox("Show Pair Plot"):
            st.subheader("Pair Plot")
            sns.pairplot(data[numeric_filter])
            st.pyplot()

        if st.checkbox("Show Box Plot"):
            box_feature = st.selectbox("Select Feature for Box Plot", numeric_filter)
            sns.boxplot(x=data[box_feature])
            st.pyplot()

        if st.checkbox("Show Violin Plot"):
            violin_feature = st.selectbox("Select Feature for Violin Plot", numeric_filter)
            sns.violinplot(x=data[violin_feature])
            st.pyplot()

    # Closing message
    st.write("### Thank you for using the Breast Cancer Analysis App!")
