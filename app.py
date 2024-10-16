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

    # Multi-select for categorical and numeric features
    categorical_filter = data.select_dtypes(include='object').columns.tolist()
    numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

    selected_categorical = st.sidebar.multiselect("Select Categorical Columns", categorical_filter)
    selected_numeric = st.sidebar.multiselect("Select Numeric Columns", numeric_filter)

    # Tab structure
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Data Overview", "Correlation Heatmap", "Imputation Comparison", "Scaling", "Visualizations"])

    # Data Overview Tab
    with tab1:
        st.subheader("Data Overview")
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
            sns.pairplot(data[selected_numeric])
            st.pyplot()

        if st.checkbox("Show Box Plot"):
            box_feature = st.selectbox("Select Feature for Box Plot", selected_numeric)
            sns.boxplot(x=data[box_feature])
            st.pyplot()

        if st.checkbox("Show Violin Plot"):
            violin_feature = st.selectbox("Select Feature for Violin Plot", selected_numeric)
            sns.violinplot(x=data[violin_feature])
            st.pyplot()

    # Closing message
    st.write("### Thank you for using the Breast Cancer Analysis App!")
