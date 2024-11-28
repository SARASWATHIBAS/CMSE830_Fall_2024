import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Load the dataset from GitHub
url = "https://raw.githubusercontent.com/SARASWATHIBAS/CMSE830_Fall_2024/main/SEER%20Breast%20Cancer%20Dataset%20.csv"

# Add plain background image
# Set background image
st.markdown(
    """
    <style>
   .stButton > button {
        background-color: #007BFF; /* Button color */
        color: white; /* Button text color */
        border: none; /* Remove border */
        border-radius: 5px; /* Rounded corners */
        padding: 10px; /* Padding */
        font-size: 16px; /* Font size */
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
# Sidebar filters
st.sidebar.header("Filter Data")


# Multi-select for categorical and numeric features
categorical_filter = data.select_dtypes(include='object').columns.tolist()
numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

# Default selections for categorical and numeric columns
default_categorical = categorical_filter[:2]  # Select the first 2 categorical columns as default
default_numeric = numeric_filter[:2]  # Select the first 2 numeric columns as default


# Store selections in session state to maintain state across runs
if 'selected_categorical' not in st.session_state:
    st.session_state.selected_categorical = []
if 'selected_numeric' not in st.session_state:
    st.session_state.selected_numeric = []
if 'is_filtered' not in st.session_state:
    st.session_state.is_filtered = False

# Allow users to reset their selections
if st.sidebar.button("Reset Filters", key="reset_filters_button"):
    st.session_state.selected_categorical = []
    st.session_state.selected_numeric = []
    st.session_state.is_filtered = False  # Reset the filter state as well

# Multi-select for categorical and numeric columns
selected_categorical = st.sidebar.multiselect(
    "Select Categorical Columns",
    categorical_filter,
    default=default_categorical,
    key="categorical_multiselect"
)
selected_numeric = st.sidebar.multiselect(
    "Select Numeric Columns",
    numeric_filter,
    default=default_numeric,
    key="numeric_multiselect"
)

# Add a "Go" button
if st.sidebar.button("Go", key="go_button"):
    # Update session state with current selections
    st.session_state.selected_categorical = selected_categorical
    st.session_state.selected_numeric = selected_numeric
    st.session_state.is_filtered = True  # Set filter state to True

# Notify user of selections
if st.session_state.is_filtered:
    st.sidebar.write("### Selected Filters:")
    st.sidebar.write(f"**Categorical Columns:** {', '.join(st.session_state.selected_categorical) if st.session_state.selected_categorical else 'None'}")
    st.sidebar.write(f"**Numeric Columns:** {', '.join(st.session_state.selected_numeric) if st.session_state.selected_numeric else 'None'}")


# Tabs for app sections
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8= st.tabs(
    ["Data Overview","Search", "Correlation Heatmap", "Imputation Comparison", "Scaling", "Visualizations", "Modeling", "Advanced Data Cleaning and Preprocessing"]
)

# Data Overview Tab
with tab1:
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


# Search Tab with Dropdown for Categorical Variables
with tab2:
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
# Modeling Tab
with tab7:
    st.subheader("Modeling Analysis")

    st.subheader("K-Means Clustering Analysis")

    # User selects features for clustering
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    selected_features = st.multiselect("Select Features for Clustering (Choose 2)",
                                       options=numerical_features, default=["Tumor Size", "Age"])

    if len(selected_features) == 2:  # Ensure exactly two features are selected
        # Extract selected features for clustering

        X_clustering = data[selected_features]

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        data['Cluster'] = kmeans.fit_predict(X_clustering)

        # Create interactive scatter plot using Plotly
        fig = px.scatter(data, x=selected_features[0], y=selected_features[1],
                         color='Cluster', title='K-Means Clustering of Selected Features',
                         labels={selected_features[0]: selected_features[0],
                                 selected_features[1]: selected_features[1]},
                         color_continuous_scale=px.colors.sequential.Viridis)

        # Update layout for better display
        fig.update_traces(marker=dict(size=10, opacity=0.6))
        fig.update_layout(legend_title_text='Cluster')

        # Display the plot in Streamlit
        st.plotly_chart(fig)

    else:
        st.warning("Please select exactly **two** features for clustering.")
    # Polynomial Regression Analysis
    if st.checkbox("Perform Polynomial Regression Analysis"):
        st.subheader("Polynomial Regression Analysis")

        # Select numerical feature
        selected_numeric_feature = st.selectbox("Choose a Numerical Feature", numeric_filter)

        # Ensure the chosen feature is valid
        if selected_numeric_feature and 'Survival Months' in data.columns:
            X = data[[selected_numeric_feature]]
            y = data['Survival Months']


            # Create polynomial features
            def create_polynomial_data(X, degree=2):
                poly = PolynomialFeatures(degree=degree)
                return poly.fit_transform(X)


            # Create polynomial features
            X_poly = create_polynomial_data(X, degree=2)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

            # Fit the polynomial regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            mse = mean_squared_error(y_test, y_pred)
            st.write(f"Mean Squared Error: {mse:.2f}")

            # Plotting true vs predicted values
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test[:, 1], y_test, color='blue', label='True Values', alpha=0.5)
            plt.scatter(X_test[:, 1], y_pred, color='red', label='Predicted Values', alpha=0.5)
            plt.title(f'True vs Predicted Values for {selected_numeric_feature} vs Survival Months')
            plt.xlabel(selected_numeric_feature)
            plt.ylabel('Survival Months')
            plt.legend()
            st.pyplot(plt)

# Tab 8: Advanced Data Cleaning and Preprocessing
with tab8:
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