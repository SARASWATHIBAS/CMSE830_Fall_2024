import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import silhouette_score
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier, XGBRegressor

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
import numpy as np
from scipy import stats
from imblearn.over_sampling import SMOTE
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from umap.umap_ import UMAP


def generate_dynamic_risk_inferences(age, tumor_size, nodes_positive, grade, risk_score):
    """Generate detailed clinical risk inferences"""
    inferences = []

    # Age-based inferences
    if age < 40:
        inferences.append("Early-onset case requiring careful genetic consideration")
    elif age > 70:
        inferences.append("Age-specific treatment modifications may be necessary")

    # Tumor size inferences
    if tumor_size < 20:
        inferences.append("Early detection favorable for treatment outcomes")
    elif tumor_size > 50:
        inferences.append("Larger tumor size may indicate need for neoadjuvant therapy")

    # Lymph node inferences
    if nodes_positive == 0:
        inferences.append("Node-negative status suggests localized disease")
    elif nodes_positive > 3:
        inferences.append("Multiple positive nodes indicate lymphatic involvement")

    # Grade-based recommendations
    grade_insights = {
        "1": "Well-differentiated tumor - favorable prognosis",
        "2": "Moderate differentiation - standard protocols indicated",
        "3": "Poor differentiation - aggressive approach recommended"
    }
    inferences.append(grade_insights[grade])

    return inferences


def enhance_survival_prediction(age, stage, treatments, survival_curve):
    """Generate comprehensive survival insights"""
    insights = {
        'short_term': [],
        'long_term': [],
        'recommendations': []
    }

    # Stage-specific insights
    stage_insights = {
        "I": "Early-stage disease with favorable outlook",
        "II": "Good prognosis with appropriate intervention",
        "III": "Local advancement requires multimodal therapy",
        "IV": "Systemic disease management priority"
    }
    insights['short_term'].append(stage_insights[stage])

    # Treatment combination analysis
    if "Surgery" in treatments and "Chemotherapy" in treatments:
        insights['recommendations'].append("Standard combination therapy on track")
    if "Radiation" in treatments and age > 70:
        insights['recommendations'].append("Consider radiation fractionation adjustment")

    # Long-term outlook
    five_year_prob = survival_curve['probability'][60]
    if five_year_prob > 0.8:
        insights['long_term'].append("Excellent long-term survival probability")
    elif five_year_prob > 0.6:
        insights['long_term'].append("Good long-term outlook with monitoring")
    else:
        insights['long_term'].append("Close surveillance and support recommended")

    return insights


def generate_treatment_insights(age, stage, comorbidities, plans):
    """Generate personalized treatment insights"""
    insights = []

    # Age-specific considerations
    if age > 75:
        insights.append("Consider treatment de-escalation based on age")

    # Comorbidity impact
    if "Diabetes" in comorbidities:
        insights.append("Monitor glucose levels during treatment")
    if "Heart Disease" in comorbidities:
        insights.append("Cardiac monitoring during therapy recommended")

    # Stage-specific approach
    if stage in ["III", "IV"]:
        insights.append("Multimodal therapy approach indicated")

    return insights


def calculate_risk_score(age, tumor_size, nodes_positive, grade):
    """Calculate comprehensive risk score based on clinical parameters"""
    # Normalize inputs
    age_factor = age / 100
    size_factor = tumor_size / 200
    node_factor = nodes_positive / 50
    grade_factor = int(grade) / 3

    # Weighted calculation
    risk_score = (
                         0.25 * age_factor +
                         0.35 * size_factor +
                         0.25 * node_factor +
                         0.15 * grade_factor
                 ) * 100

    return risk_score


def predict_survival(age, stage, treatments):
    """Generate survival predictions based on patient characteristics"""
    # Base survival rate by stage
    stage_rates = {
        "I": 0.95,
        "II": 0.85,
        "III": 0.70,
        "IV": 0.45
    }

    # Treatment impact factors
    treatment_factors = {
        "Surgery": 0.15,
        "Chemotherapy": 0.12,
        "Radiation": 0.10,
        "Hormone Therapy": 0.08
    }

    # Calculate survival probability
    base_rate = stage_rates[stage]
    treatment_boost = sum(treatment_factors[t] for t in treatments)
    age_factor = (100 - age) / 100

    survival_rate = min(1.0, base_rate + treatment_boost) * age_factor

    return generate_survival_curve(survival_rate)


def generate_survival_curve(survival_rate):
    """Generate 5-year survival curve data"""
    months = range(0, 61)
    survival_probs = [survival_rate * (1 - month / 120) for month in months]
    return {'months': months, 'probability': survival_probs}


def plot_survival_curve(survival_data):
    """Create advanced interactive survival curve visualization"""
    fig = go.Figure()

    # Convert data
    x_values = list(survival_data['months'])
    y_values = survival_data['probability']

    # Main survival curve
    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode='lines',
        name='Survival Probability',
        line=dict(color='#2E86C1', width=3)
    ))

    # Add confidence intervals
    upper_ci = [min(1, p + 0.1) for p in y_values]
    lower_ci = [max(0, p - 0.1) for p in y_values]

    fig.add_trace(go.Scatter(
        x=x_values + x_values[::-1],
        y=upper_ci + lower_ci[::-1],
        fill='toself',
        fillcolor='rgba(46, 134, 193, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))

    # Add risk stratification lines
    fig.add_trace(go.Scatter(
        x=x_values,
        y=[0.75] * len(x_values),
        mode='lines',
        line=dict(color='green', dash='dash'),
        name='High Survival'
    ))

    fig.add_trace(go.Scatter(
        x=x_values,
        y=[0.5] * len(x_values),
        mode='lines',
        line=dict(color='orange', dash='dash'),
        name='Moderate Risk'
    ))

    # Add milestone markers
    milestones = {
        12: '1 Year',
        36: '3 Years',
        60: '5 Years'
    }

    milestone_x = []
    milestone_y = []
    milestone_text = []

    for month, label in milestones.items():
        milestone_x.append(month)
        milestone_y.append(y_values[month])
        milestone_text.append(label)

    fig.add_trace(go.Scatter(
        x=milestone_x,
        y=milestone_y,
        mode='markers+text',
        marker=dict(size=10, symbol='diamond', color='#2E86C1'),
        text=milestone_text,
        textposition='top center',
        name='Key Milestones'
    ))

    # Enhanced layout
    fig.update_layout(
        title={
            'text': '5-Year Survival Probability Analysis',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title='Months',
        yaxis_title='Survival Probability',
        yaxis=dict(
            tickformat='.0%',
            range=[0, 1]
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        plot_bgcolor='white',
        width=800,
        height=500
    )

    # Add gridlines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    st.plotly_chart(fig)

def display_survival_metrics(survival_data):
    """Display key survival statistics"""
    col1, col2, col3 = st.columns(3)
    col1.metric("1-Year Survival", f"{survival_data['probability'][12]:.1%}")
    col2.metric("3-Year Survival", f"{survival_data['probability'][36]:.1%}")
    col3.metric("5-Year Survival", f"{survival_data['probability'][60]:.1%}")


def generate_treatment_plans(age, stage, comorbidities):
    """Generate personalized treatment plans"""
    base_plans = {
        "I": ["Surgery", "Radiation"],
        "II": ["Surgery", "Chemotherapy", "Radiation"],
        "III": ["Chemotherapy", "Surgery", "Radiation"],
        "IV": ["Chemotherapy", "Targeted Therapy", "Palliative Care"]
    }

    # Adjust for age and comorbidities
    plans = []
    base_treatments = base_plans[stage]

    # Generate multiple plan options
    plans.append({
        'name': 'Standard Protocol',
        'treatments': base_treatments,
        'effectiveness': calculate_effectiveness(base_treatments, age, comorbidities),
        'risk_level': 'Moderate'
    })

    # Conservative option
    conservative = [t for t in base_treatments if t not in ['Chemotherapy']]
    plans.append({
        'name': 'Conservative Approach',
        'treatments': conservative,
        'effectiveness': calculate_effectiveness(conservative, age, comorbidities),
        'risk_level': 'Low'
    })

    return plans


def calculate_effectiveness(treatments, age, comorbidities):
    """Calculate treatment effectiveness score"""
    base_score = len(treatments) * 20
    age_factor = (100 - age) / 100
    comorbidity_factor = 1 - (len(comorbidities) * 0.1)

    return base_score * age_factor * comorbidity_factor


def display_treatment_options(plans):
    """Display treatment plans with interactive elements"""
    for i, plan in enumerate(plans):
        with st.expander(f"Treatment Plan {i + 1}: {plan['name']}"):
            st.write("### Treatments")
            for treatment in plan['treatments']:
                st.write(f"- {treatment}")

            st.write("### Metrics")
            col1, col2 = st.columns(2)
            col1.metric("Effectiveness Score", f"{plan['effectiveness']:.1f}")
            col2.metric("Risk Level", plan['risk_level'])

            st.write("### Additional Considerations")
            st.write(get_treatment_considerations(plan))


def get_treatment_considerations(plan):
    """Generate treatment-specific considerations"""
    considerations = {
        'Standard Protocol': "Balanced approach with proven outcomes",
        'Conservative Approach': "Minimizes treatment intensity while maintaining efficacy"
    }
    return considerations[plan['name']]


def run_risk_assessment():
    st.subheader("Patient Risk Calculator")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Patient Age", 18, 100, key="risk_age")
        tumor_size = st.number_input("Tumor Size (mm)", 0.0, 200.0, key="risk_size")
        nodes_positive = st.number_input("Positive Lymph Nodes", 0, 50, key="risk_nodes")

    with col2:
        grade = st.selectbox("Tumor Grade", ["1", "2", "3"], key="risk_grade")
        stage = st.selectbox("Cancer Stage", ["I", "II", "III", "IV"], key="risk_stage")
        er_status = st.selectbox("ER Status", ["Positive", "Negative"], key="risk_er")

    if st.button("Calculate Risk Score", key="risk_button"):
        risk_score = calculate_risk_score(age, tumor_size, nodes_positive, grade)
        st.metric("Risk Score", f"{risk_score:.2f}")

        # Dynamic Risk Inferences
        st.write("### Clinical Risk Assessment")
        risk_inferences = {
            'High Risk Factors': [],
            'Protective Factors': [],
            'Recommendations': []
        }

        # Age analysis
        if age < 40:
            risk_inferences['High Risk Factors'].append("Early onset indicates possible genetic factors")
        elif age > 70:
            risk_inferences['Recommendations'].append("Consider age-adjusted treatment protocols")

        # Tumor characteristics
        if tumor_size > 50:
            risk_inferences['High Risk Factors'].append("Large tumor size suggests advanced disease")
        else:
            risk_inferences['Protective Factors'].append("Tumor size within manageable range")

        # Display inferences
        for category, insights in risk_inferences.items():
            if insights:
                st.write(f"**{category}:**")
                for insight in insights:
                    st.write(f"• {insight}")


def run_survival_prediction():
    st.subheader("Survival Prediction Tool")

    age = st.number_input("Age", 18, 100, 50, key="survival_age")
    stage = st.selectbox("Disease Stage", ["I", "II", "III", "IV"], key="survival_stage")
    treatment = st.multiselect(
        "Selected Treatments",
        ["Surgery", "Chemotherapy", "Radiation", "Hormone Therapy"],
        key="survival_treatments"
    )

    if st.button("Generate Prediction", key="survival_button"):
        survival_curve = predict_survival(age, stage, treatment)
        plot_survival_curve(survival_curve)
        display_survival_metrics(survival_curve)

        # Dynamic Survival Inferences
        st.write("### Survival Analysis Insights")
        survival_insights = {
            'Prognosis Factors': [],
            'Treatment Impact': [],
            'Monitoring Recommendations': []
        }

        # Stage-based analysis
        stage_insights = {
            "I": "Early stage with excellent prognosis",
            "II": "Good prognosis with appropriate intervention",
            "III": "Moderate prognosis requiring aggressive treatment",
            "IV": "Guarded prognosis - focus on quality of life"
        }
        survival_insights['Prognosis Factors'].append(stage_insights[stage])

        # Treatment analysis
        if len(treatment) >= 3:
            survival_insights['Treatment Impact'].append("Comprehensive treatment approach selected")

        # Display insights
        for category, insights in survival_insights.items():
            if insights:
                st.write(f"**{category}:**")
                for insight in insights:
                    st.write(f"• {insight}")


def run_treatment_planning():
    st.subheader("Treatment Planning Assistant")

    st.write("### Patient Profile")
    age = st.number_input("Age", 18, 100, 50, key="treatment_age")
    stage = st.selectbox("Stage", ["I", "II", "III", "IV"], key="treatment_stage")
    comorbidities = st.multiselect(
        "Comorbidities",
        ["Diabetes", "Hypertension", "Heart Disease", "None"],
        key="treatment_comorbidities"
    )

    if st.button("Generate Treatment Plans", key="treatment_button"):
        plans = generate_treatment_plans(age, stage, comorbidities)
        display_treatment_options(plans)

        # Dynamic Treatment Inferences
        st.write("### Treatment Strategy Insights")
        treatment_insights = {
            'Key Considerations': [],
            'Risk Factors': [],
            'Optimization Suggestions': []
        }

        # Age-based considerations
        if age > 75:
            treatment_insights['Key Considerations'].append("Consider reduced treatment intensity")

        # Comorbidity analysis
        for comorbidity in comorbidities:
            if comorbidity != "None":
                treatment_insights['Risk Factors'].append(f"Monitor {comorbidity.lower()} during treatment")

        # Stage-based recommendations
        if stage in ["III", "IV"]:
            treatment_insights['Optimization Suggestions'].append("Consider clinical trial eligibility")

        # Display insights
        for category, insights in treatment_insights.items():
            if insights:
                st.write(f"**{category}:**")
                for insight in insights:
                    st.write(f"• {insight}")


def display_input_requirements():
    st.markdown("""
    ## Input Requirements

    ### Clinical Data Format
    | Field | Type | Valid Range | Required |
    |-------|------|-------------|----------|
    | Age | Integer | 18-100 | Yes |
    | Tumor Size | Float | 0.1-200.0 mm | Yes |
    | Grade | Integer | 1-3 | Yes |
    | Lymph Nodes | Integer | 0-50 | Yes |
    | ER Status | String | Positive/Negative | Yes |

    ### Data Quality Standards
    - Complete patient records
    - Validated clinical measurements
    - Standardized units
    - Recent test results (< 30 days)
    """)


def display_expected_outputs():
    st.markdown("""
    ## Expected Outputs

    ### 1. Risk Assessment
    - Numerical risk score (0-100)
    - Risk category classification
    - Confidence intervals

    ### 2. Survival Analysis
    - 5-year survival probability
    - Personalized survival curve
    - Key milestone predictions
    - Risk-adjusted outcomes

    ### 3. Treatment Recommendations
    - Ranked treatment options
    - Expected effectiveness scores

    """)


def create_radius_plot(data, selected_features, hue_feature=None):
    """
    Creates an interactive radius plot showing feature relationships
    """
    # Normalize the features for better visualization
    normalized_data = pd.DataFrame()
    for feature in selected_features:
        normalized_data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())

    # Create the radius plot
    fig = go.Figure()

    if hue_feature:
        for category in data[hue_feature].unique():
            mask = data[hue_feature] == category
            values = normalized_data[mask][selected_features].mean()

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=selected_features,
                name=str(category),
                fill='toself'
            ))
    else:
        values = normalized_data[selected_features].mean()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=selected_features,
            fill='toself'
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Feature Radius Plot"
    )

    return fig


def interpret_correlation(correlation, feature_x, feature_y):
    """Provides detailed correlation interpretation"""
    st.write(f"### Correlation Analysis: {feature_x} vs {feature_y}")

    # Correlation strength interpretation
    if abs(correlation) >= 0.8:
        strength = "Very Strong"
    elif abs(correlation) >= 0.6:
        strength = "Strong"
    elif abs(correlation) >= 0.4:
        strength = "Moderate"
    elif abs(correlation) >= 0.2:
        strength = "Weak"
    else:
        strength = "Very Weak"

    # Direction interpretation
    direction = "Positive" if correlation > 0 else "Negative"

    st.write(f"""
    #### Strength: {strength} {direction} Correlation ({correlation:.3f})

    Key Insights:
    - The relationship between these features is {strength.lower()}
    - As {feature_x} increases, {feature_y} tends to {'increase' if correlation > 0 else 'decrease'}
    - This correlation explains {(correlation ** 2 * 100):.1f}% of the variance
    """)


def create_violin_plot(data, numerical_feature, categorical_feature, hue_feature):
    """Creates an enhanced violin plot with statistical annotations"""
    fig = plt.figure(figsize=(12, 6))

    # Create violin plot with swarm overlay
    sns.violinplot(data=data, x=categorical_feature, y=numerical_feature,
                   hue=hue_feature, split=True)
    sns.swarmplot(data=data, x=categorical_feature, y=numerical_feature,
                  color="white", alpha=0.5, size=4)

    plt.title(f'Distribution of {numerical_feature} by {categorical_feature}')
    plt.xticks(rotation=45)

    # Add statistical annotations
    categories = data[categorical_feature].unique()
    means = data.groupby(categorical_feature)[numerical_feature].mean()

    for i, cat in enumerate(categories):
        plt.text(i, means[cat], f'μ={means[cat]:.2f}',
                 horizontalalignment='center', verticalalignment='bottom')

    st.pyplot(fig)
    plt.close()


def create_box_plot(data, numerical_feature, categorical_feature, hue_feature):
    """Creates an enhanced box plot with statistical insights"""
    fig = plt.figure(figsize=(12, 6))

    # Create box plot with points
    sns.boxplot(data=data, x=categorical_feature, y=numerical_feature,
                hue=hue_feature, showfliers=True)

    plt.title(f'Box Plot: {numerical_feature} by {categorical_feature}')
    plt.xticks(rotation=45)

    # Add statistical summary
    stats = data.groupby(categorical_feature)[numerical_feature].describe()
    st.write("### Statistical Summary")
    st.dataframe(stats)

    st.pyplot(fig)
    plt.close()


def create_pair_plot(data, selected_numeric, hue_feature):
    """Creates an enhanced pair plot with customizable features"""
    # Limit to manageable number of features for performance
    if len(selected_numeric) > 5:
        st.warning("Limiting to first 5 numeric features for clarity")
        selected_numeric = selected_numeric[:5]

    fig = sns.pairplot(data[selected_numeric + [hue_feature]],
                       hue=hue_feature,
                       diag_kind="kde",
                       plot_kws={'alpha': 0.6},
                       height=2.5)

    fig.fig.suptitle("Pair-wise Feature Relationships", y=1.02)

    # Add correlation coefficients
    for i in range(len(selected_numeric)):
        for j in range(len(selected_numeric)):
            if i != j:
                corr = data[selected_numeric[i]].corr(data[selected_numeric[j]])
                fig.axes[i, j].annotate(f'ρ={corr:.2f}',
                                        xy=(0.5, 0.9),
                                        xycoords='axes fraction',
                                        ha='center')

    st.pyplot(fig)
    plt.close()


def get_visualization_insights(viz_type, feature_x, feature_y, dist_type="Box Plot", plot_3d_type="Scatter",z_feature=None):
    insights = {
        "Scatter Plot Analysis": f"""
        #### Key Insights for {feature_x} vs {feature_y}:
        1. The scatter pattern between {feature_x} and {feature_y} reveals their relationship strength and direction.
        2. Clustered points in the {feature_x}-{feature_y} space indicate potential subgroups worth investigating.
        """,

        "Violin": f"""
        #### Distribution Insights for {feature_x}:
        1. The violin width shows the concentration of {feature_x} values, highlighting common ranges.
        2. Shape variations in {feature_x} distribution indicate potential data patterns and outliers.
        """,

        "Box Plot": f"""
        #### Box Plot Insights for {feature_x}:
        1. The quartile boundaries for {feature_x} show its spread and central tendency.
        2. Points beyond {feature_x}'s whiskers represent unique cases requiring attention.
        """,

        "Pair Plot": f"""
        #### Pair Plot Insights for Selected Features:
        1. The diagonal distributions show how {feature_x} and {feature_y} are individually distributed.
        2. The scatter matrices reveal relationships between {feature_x}, {feature_y} and other variables.
        """,

        "3D Scatter": f"""
        #### 3D Scatter Insights for {feature_x}, {feature_y}, and {z_feature}:
        1. The spatial arrangement shows how {feature_x}, {feature_y}, and {z_feature} interact.
        2. Color patterns highlight groupings across these three dimensions.
        """,

        "Surface": f"""
        #### Surface Plot Insights for {feature_x}, {feature_y}, and {z_feature}:
        1. Surface heights show how {z_feature} varies with changes in {feature_x} and {feature_y}.
        2. Color intensities highlight key regions in the {feature_x}-{feature_y} plane.
        """,

        "Radius": f"""
        #### Radius Plot Insights for Selected Features:
        1. The shape symmetry shows balance across {feature_x}, {feature_y} and other selected features.
        2. Pattern overlaps indicate relationships between feature groups.
        """
    }

    # Return appropriate insight based on visualization type
    if viz_type == "Scatter Plot Analysis" or viz_type == "All Plots":
        return insights["Scatter Plot Analysis"]
    elif viz_type == "Distribution Plots":
        return insights.get(dist_type, insights["Box Plot"])
    elif viz_type == "3D Visualizations":
        return insights["3D Scatter"] if plot_3d_type == "Scatter" else insights["Surface"]
    elif viz_type == "Radius Plot":
        return insights["Radius"]


def production_space():
    def show_documentation():
        """Display comprehensive documentation and user guide"""
        with st.expander("📚 Documentation & User Guide", expanded=False):
            st.markdown("""
            # Breast Cancer Analysis App Documentation

            ## Overview
            This application provides comprehensive tools for analyzing breast cancer data through various statistical and machine learning approaches.
            
            ## Purpose
            The Production space is designed to address the challenge of predicting the likelihood of a person being diagnosed with breast cancer based on their specific characteristics. It also aims to raise awareness and provide foundational guidance to clinical professionals on potential steps for effective treatment planning.
            ## Key Features
            1. **Data Analysis**
               - Data overview and statistics
               - Missing value analysis
               - Correlation studies

            2. **Visualization Tools**
               - Interactive plots
               - Statistical visualizations
               - Distribution analysis

            3. **Machine Learning**
               - Classification models
               - Clustering analysis
               - Regression predictions

            4. **Data Processing**
               - Advanced cleaning
               - Feature engineering
               - Dimensionality reduction

            ## How to Use

            ### 1. Data Selection
            - Use the sidebar to select features
            - Choose categorical and numerical columns
            - Apply filters as needed

            ### 2. Analysis Workflow
            1. Start with Data Overview
            2. Perform initial visualizations
            3. Apply preprocessing steps
            4. Run machine learning models

            ### 3. Tips for Best Results
            - Select relevant features for analysis
            - Check data quality before modeling
            - Use appropriate scaling methods

            ## Tab Guide

            1. **Data Overview**: Basic statistics and data summary
            2. **Search**: Find specific data points
            3. **Correlation**: Analyze feature relationships
            4. **Imputation**: Handle missing values
            5. **Scaling**: Normalize data
            6. **Visualizations**: Create plots
            7. **Modeling**: Build ML models
            8. **Advanced Cleaning**: Deep data preprocessing
            9. **Advanced Analysis**: Complex analytical tools
            10. **Feature Engineering**: Create new features

            ## Best Practices
            - Always check data quality first
            - Use appropriate visualization for your data type
            - Consider feature relationships before modeling
            """)
    st.header("Clinical Production Space")


    tool_choice = st.selectbox(
        "Select Clinical Tool",
        ["User Guide","Risk Assessment", "Survival Prediction", "Treatment Planning"]
    )

    if tool_choice == "User Guide":
        show_documentation()
        display_input_requirements()

        display_expected_outputs()


    if tool_choice == "Risk Assessment":
        run_risk_assessment()
    elif tool_choice == "Survival Prediction":
        run_survival_prediction()
    elif tool_choice == "Treatment Planning":
        run_treatment_planning()


def create_3d_scatter(data, x_col, y_col, z_col, color_col):
    """
    Creates an interactive 3D scatter plot with formatted labels
    """
    # Create formatted labels dictionary
    labels = {
        x_col: x_col,
        y_col: y_col,
        z_col: z_col
    }

    fig = px.scatter_3d(
        data,
        x=x_col,
        y=y_col,
        z=z_col,
        color=color_col,
        opacity=0.7,
        title=f"3D Scatter Plot: {x_col} vs {y_col} vs {z_col}",
        labels=labels
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        width=800,
        height=800
    )

    return fig


def create_3d_surface(data, x_col, y_col, z_col):
    """
    Creates an interactive 3D surface plot
    """
    # Create a pivot table for surface plotting
    pivot_data = data.pivot_table(
        values=z_col,
        index=x_col,
        columns=y_col,
        aggfunc='mean'
    )

    fig = go.Figure(data=[go.Surface(z=pivot_data.values)])

    fig.update_layout(
        title=f'3D Surface Plot: {z_col} by {x_col} and {y_col}',
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        width=800,
        height=800
    )

    return fig



def data_science_space():
    st.header("Data Science Research Space")


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
        .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                overflow-x: scroll;
                scrollbar-width: thin;
                padding: 5px 5px;
                margin-bottom: 10px;
            }

            .stTabs [data-baseweb="tab"] {
                min-width: 300px    ;
                font-size: 14px;
                padding: 5px 10px;
                background-color: #f0f2f6;
                border-radius: 4px;
                margin-right: 5px;
            }

            .stTabs [data-baseweb="tab"]:hover {
                background-color: #e0e2e6;
            }

            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
                height: 6px;
            }

            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-track {
                background: #f1f1f1;
                border-radius: 4px;
            }

            .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
                background: #888;
                border-radius: 4px;

            }

        </style>
        """,

        unsafe_allow_html=True
    )
    # Data Caching
    @st.cache_data
    def cache_data(url):
        """Cache the initial dataset loading"""
        return pd.read_csv(url)

    def cache_processed_features(data):
        """
        Cache and return processed numeric and categorical features

        Parameters:
            data (pd.DataFrame): Input dataset

        Returns:
            tuple: (numeric_columns, categorical_columns)
        """
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()

        # Get categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

        # Add additional feature processing if needed
        # Example: Remove specific columns, rename columns, etc.

        return numeric_cols, categorical_cols

    # Usage in your app:
    data = cache_data(url)

    # Remove any unnamed columns
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    numeric_cols, categorical_cols = cache_processed_features(data)

    # Sidebar filters
    st.sidebar.header("Filter Data")
    data_actual = data.copy()

    # Multi-select for categorical and numeric features
    categorical_filter = data.select_dtypes(include='object').columns.tolist()
    numeric_filter = data.select_dtypes(include=np.number).columns.tolist()

    # Default selections for categorical and numeric columns
    default_categorical = categorical_filter[:2]  # Select the first 2 categorical columns as default
    default_numeric = numeric_filter[:3]  # Select the first 2 numeric columns as default

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
        st.sidebar.write(
            f"**Categorical Columns:** {', '.join(st.session_state.selected_categorical) if st.session_state.selected_categorical else 'None'}")
        st.sidebar.write(
            f"**Numeric Columns:** {', '.join(st.session_state.selected_numeric) if st.session_state.selected_numeric else 'None'}")

    # Create tabs with descriptive names
    tabs = st.tabs([
        "Data Overview", "Search","Advanced Data Cleaning Preprocessing", "Visualizations","Advanced Data Analysis", "Data Processing & Feature Engineering",
        "Modeling","Technical Documentation"

    ])

    # Assign tabs to variables
    tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8 = tabs

    # Use a session state to store the active tab
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = tab1

    # Data Overview Tab
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
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
        - **reginol Node Examined/Positive:** Number of nodes examined and found positive.
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
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("Search Data")

        # Select column to search within
        search_column = st.selectbox("Select Column to Search", data.columns)

        # Different search interfaces based on column type
        if search_column in categorical_filter:
            # Categorical search with dropdown
            search_value = st.selectbox("Select Value", data[search_column].unique())

            # Filter data for categorical columns
            filtered_data = data[data[search_column] == search_value]

            # Display results immediately for categorical selection
            st.write(f"### Results for: **{search_value}** in **{search_column}**")
            st.write(filtered_data)

        else:
            # Numeric/text search with manual input
            search_value = st.text_input("Enter Search Value")

            if st.button("Search"):
                filtered_data = data[
                    data[search_column].astype(str).str.contains(str(search_value), case=False, na=False)]


            # Common display section for both types
            if 'filtered_data' in locals() and not filtered_data.empty:
                # Dynamic Inferences Section
                st.write("### Data Insights")

                # Row count analysis
                total_rows = len(data)
                filtered_rows = len(filtered_data)
                percentage = (filtered_rows / total_rows) * 100

                st.write(f"#### Sample Size Analysis")
                st.write(f"• Found {filtered_rows} matching records ({percentage:.1f}% of total data)")

                if percentage > 50:
                    st.write("• This represents a majority of the dataset")
                elif percentage < 10:
                    st.write("• This represents a small subset of the dataset")

                # Value distribution insights
                if filtered_rows > 1:
                    numeric_cols = filtered_data.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        st.write("#### Numeric Patterns")
                        for col in numeric_cols:
                            mean_val = filtered_data[col].mean()
                            overall_mean = data[col].mean()

                            if mean_val > overall_mean:
                                st.write(f"• {col}: Values are higher than dataset average")
                            else:
                                st.write(f"• {col}: Values are lower than dataset average")

                # Display basic statistics
                st.write("### Summary Statistics")
                st.write(filtered_data.describe(include='all'))

                # Additional insights for numeric columns
                numeric_cols = filtered_data.select_dtypes(include=np.number).columns
                if len(numeric_cols) > 0:
                    st.write("#### Numeric Statistics")
                    stats_df = filtered_data[numeric_cols].agg(['mean', 'median', 'min', 'max', 'count'])
                    st.write(stats_df)

                    # Statistical significance insights
                    st.write("#### Statistical Significance")
                    for col in numeric_cols:
                        overall_std = data[col].std()
                        filtered_mean = filtered_data[col].mean()
                        overall_mean = data[col].mean()

                        if abs(filtered_mean - overall_mean) > overall_std:
                            st.write(f"• {col} shows significant deviation from overall pattern")

            elif 'filtered_data' in locals():
                st.warning("No results found. Try adjusting your search criteria.")

        st.write("Thank you for using the Breast Cancer Analysis App!")
        # Tab 3: Advanced Data Cleaning and Preprocessing
        with tab3:
            st.markdown('<div class="tab-content">', unsafe_allow_html=True)
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

            # Analyze data distribution to recommend imputation method
            numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
            skewness = data[numeric_cols].skew()

            st.write("#### Recommended Imputation Strategy:")
            for col in numeric_cols:
                if abs(skewness[col]) > 1:
                    st.write(f"• {col}: Consider median imputation due to skewed distribution")
                else:
                    st.write(f"• {col}: Mean imputation suitable due to normal distribution")

            # Column selection for imputation
            columns_to_impute = st.multiselect(
                "Select columns for imputation:",
                numeric_filter,
                default=numeric_filter[:2]
            )

            if columns_to_impute:
                imputation_method = st.radio(
                    "Select an imputation method for missing values:",
                    ["Mean Imputation", "KNN Imputation", "Drop Rows"]
                )

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                if imputation_method == "Mean Imputation":
                    mean_imputer = SimpleImputer(strategy='mean')
                    data_imputed = data.copy()
                    data_imputed[columns_to_impute] = mean_imputer.fit_transform(data[columns_to_impute])

                    # Create separate plots for each column
                    for col in columns_to_impute:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(data_imputed[col], kde=True, color='skyblue')
                        plt.title(f'Mean Imputed Distribution: {col}')
                        st.pyplot(fig)
                        plt.close()

                    st.write("Missing values filled using column means")

                elif imputation_method == "KNN Imputation":
                    knn_imputer = KNNImputer(n_neighbors=5)
                    data_imputed = data.copy()
                    data_imputed[columns_to_impute] = knn_imputer.fit_transform(data[columns_to_impute])

                    # Create separate plots for each column
                    for col in columns_to_impute:
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.histplot(data_imputed[col], kde=True, color='salmon')
                        plt.title(f'KNN Imputed Distribution: {col}')
                        st.pyplot(fig)
                        plt.close()

                    st.write("Missing values filled using KNN Imputation")

                elif imputation_method == "Drop Rows":
                    data_imputed = data.dropna(subset=columns_to_impute)
                    st.write(f"Rows with missing values in selected columns dropped")
                    # Display imputation results
                    st.pyplot(fig)

                st.write("### Cleaned Data Preview")
                st.write(data_imputed[columns_to_impute].head())

            # Encoding Categorical Variables
            categorical_cols = data.select_dtypes(include=['object']).columns

            if len(categorical_cols) > 0:
                st.write("#### Cardinality Analysis:")
                for col in categorical_cols:
                    unique_values = data[col].nunique()
                    if unique_values > 10:
                        st.write(
                            f"• {col}: High cardinality ({unique_values} unique values) - Consider target encoding")
                    else:
                        st.write(f"• {col}: Suitable for one-hot encoding ({unique_values} unique values)")
            encoding_method = st.selectbox("Choose encoding method", ("Label Encoding", "One-Hot Encoding"))
            if encoding_method == "Label Encoding":
                label_column = st.selectbox("Select column for Label Encoding",
                                            data.select_dtypes(include=['object']).columns)
                label_encoder = LabelEncoder()
                data_encoded = data.copy()
                data_encoded[label_column] = label_encoder.fit_transform(data[label_column])
                st.write(f"Label Encoded Data for {label_column}:", data_encoded.head())
            elif encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns)
                st.write("One-Hot Encoded Data:", data.head())

            # Normalization and Scaling
            numeric_summary = data[numeric_cols].describe()

            for col in numeric_cols:
                range_val = numeric_summary.loc['max', col] - numeric_summary.loc['min', col]
                std_val = numeric_summary.loc['std', col]

                st.write(f"#### {col} Scaling Recommendation:")
                if range_val > 100:
                    st.write("• Consider Min-Max scaling due to large value range")
                if std_val > 10:
                    st.write("• Consider Standardization due to high standard deviation")
                if abs(skewness[col]) > 2:
                    st.write("• Consider Robust scaling due to significant outliers")
            scale_method = st.selectbox("Choose scaling method",
                                        ("Min-Max Scaling", "Standardization", "Robust Scaling"))
            if scale_method == "Min-Max Scaling":
                scaler = MinMaxScaler()
                data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
                data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
                st.write("Min-Max Scaled Data:", data.head())
            elif scale_method == "Standardization":
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
                data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
                st.write("Standardized Data:", data.head())
            elif scale_method == "Robust Scaling":
                scaler = RobustScaler()
                data_scaled = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
                data[data.select_dtypes(include=['float64', 'int64']).columns] = data_scaled
                st.write("Robust Scaled Data:", data.head())

            # Feature Engineering: Extracting Date-Time Features
            if "date" in data.columns:
                data['year'] = pd.to_datetime(data['date']).dt.year
                data['month'] = pd.to_datetime(data['date']).dt.month
                data['weekday'] = pd.to_datetime(data['date']).dt.weekday
                st.write("Extracted Date Features:", data.head())

            # Binning continuous features (e.g., age)
            if "age" in data.columns:
                data['age_group'] = pd.cut(data['age'], bins=[0, 18, 35, 50, 100],
                                           labels=['0-18', '19-35', '36-50', '51+'])
                st.write("Binned Age Groups:", data.head())

            # Handling Outliers: Z-Score and IQR Methods
            # Outlier Analysis with Dynamic Insights
            st.write("### Outlier Analysis")
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outliers_count = len(z_scores[z_scores > 3])
                if outliers_count > 0:
                    percentage = (outliers_count / len(data)) * 100
                    st.write(f"• {col}: {outliers_count} outliers detected ({percentage:.1f}% of data)")
                    if percentage > 5:
                        st.write("  - Consider investigating these outliers before removal")
            outlier_method = st.selectbox("Choose Outlier Detection Method", ("Z-Score Method", "IQR Method"))

            if outlier_method == "Z-Score Method":
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns

                if len(numeric_columns) == 0:
                    st.error("No numeric columns available for Z-score calculation.")
                else:
                    # Handle missing values
                    if data.isnull().sum().any():
                        st.warning("Data contains missing values. Proceeding to handle them.")
                        data = data.dropna()  # You can choose to fill NaN values if needed

                    # Z-Score Outlier Removal
                    z_scores = stats.zscore(data[numeric_columns])
                    abs_z_scores = np.abs(z_scores)
                    data_cleaned = data[(abs_z_scores < 3).all(axis=1)]  # Removing rows with z-score > 3
                    st.write("Data after Z-Score Outlier Removal:", data_cleaned.head())

            elif outlier_method == "IQR Method":
                numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
                Q1 = data[numeric_columns].quantile(0.25)
                Q3 = data[numeric_columns].quantile(0.75)
                IQR = Q3 - Q1
                filtered_data = data[numeric_columns][
                    ~((data[numeric_columns] < (Q1 - 1.5 * IQR)) | (data[numeric_columns] > (Q3 + 1.5 * IQR))).any(
                        axis=1)]
                st.write("Data after IQR Outlier Removal:", filtered_data.head())

            # Handling Imbalanced Data: SMOTE

            imbalanced = st.checkbox("Apply SMOTE to Handle Imbalanced Data")

            if imbalanced:
                if 'target' in data.columns:
                    st.write("### Class Balance Analysis")
                    class_distribution = data['target'].value_counts(normalize=True) * 100
                    if max(class_distribution) > 70:
                        st.write("• Significant class imbalance detected")
                        st.write(f"• Majority class: {class_distribution.index[0]} ({class_distribution.iloc[0]:.1f}%)")
                        st.write("• Consider using SMOTE or class weights")
                num_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=3)
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(data_scaled)

                # Add cluster labels to the dataset
                data["Cluster"] = cluster_labels
                st.write("Data with Cluster Labels:", data.head())

                # Balance clusters (if needed)
                cluster_counts = data["Cluster"].value_counts()
                st.write("Cluster Counts Before Balancing:", cluster_counts)

                # Resampling logic: Duplicate rows from smaller clusters
                max_cluster_size = cluster_counts.max()
                balanced_data = pd.concat(
                    [data[data["Cluster"] == cluster].sample(max_cluster_size, replace=True, random_state=42)
                     for cluster in data["Cluster"].unique()],
                    axis=0
                )

                st.write("Balanced Data After Resampling:")
                st.write(balanced_data)

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

    # Advanced Visualizations Tab
    with tab4:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.subheader("Advanced Visualizations")

        # Main visualization selector
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Scatter Plot Analysis",
             "Distribution Plots",
             "Correlation Analysis",
             "3D Visualizations",
             "Radius Plot",
             "All Plots"]
        )

        # Base feature selection
        feature_x = st.selectbox("Select Feature X", selected_numeric, key="feature_x")
        remaining_features_y = [col for col in selected_numeric if col != feature_x]
        feature_y = st.selectbox("Select Feature Y", remaining_features_y, key="feature_y")
        hue_feature = st.selectbox("Select Hue (Categorical)", selected_categorical, key="hue_feature")

        if viz_type == "Scatter Plot Analysis" or viz_type == "All Plots":
            st.subheader("Scatter Plot Analysis")
            # Scatter plot with correlation
            if feature_x and feature_y:
                fig = px.scatter(data, x=feature_x, y=feature_y,
                                 color=data[hue_feature],
                                 title="Interactive Scatter Plot")
                st.plotly_chart(fig)

                correlation = data[feature_x].corr(data[feature_y])
                st.metric("Correlation Coefficient", f"{correlation:.2f}")

                # Dynamic interpretation
                interpret_correlation(correlation, feature_x, feature_y)
                # After each visualization
                insights = get_visualization_insights(
                    viz_type=viz_type,
                    feature_x=feature_x,
                    feature_y=feature_y
                )
                st.markdown(insights)

        if viz_type == "Distribution Plots" or viz_type == "All Plots":
            st.subheader("Distribution Analysis")

            dist_type = st.selectbox("Select Distribution Plot Type",
                                     ["Violin Plot", "Box Plot", "Pair Plot"])

            numerical_feature = st.selectbox("Select Numerical Feature", numeric_filter)
            categorical_feature = st.selectbox("Select Categorical Feature", categorical_filter)

            if dist_type == "Violin Plot":
                create_violin_plot(data, numerical_feature, categorical_feature, hue_feature)
            elif dist_type == "Box Plot":
                create_box_plot(data, numerical_feature, categorical_feature, hue_feature)
            elif dist_type == "Pair Plot":
                create_pair_plot(data, selected_numeric, hue_feature)
                # After each visualization
                insights = get_visualization_insights(
                    viz_type=viz_type,

                    feature_x=feature_x,
                    feature_y=feature_y,
                    dist_type=dist_type if 'dist_type' in locals() else None,
                )
                st.markdown(insights)

        if viz_type == "Correlation Analysis" or viz_type == "All Plots":
            st.subheader("Correlation Analysis")

            if st.checkbox("Show Joint Plot"):
                joint_fig = px.density_heatmap(data, x=feature_x, y=feature_y,
                                               marginal_x="box", marginal_y="violin",
                                               title="Joint Distribution Plot")
                st.plotly_chart(joint_fig)

        if viz_type == "3D Visualizations" or viz_type == "All Plots":
            st.subheader("3D Analysis")

            z_feature = st.selectbox("Select Z Feature",
                                     [col for col in selected_numeric
                                      if col not in [feature_x, feature_y]])

            plot_3d_type = st.radio("Select 3D Plot Type", ["Scatter", "Surface"])

            if plot_3d_type == "Scatter":
                fig_3d = create_3d_scatter(data, feature_x, feature_y, z_feature,
                                           hue_feature)
                st.plotly_chart(fig_3d)
            else:
                fig_surface = create_3d_surface(data, feature_x, feature_y, z_feature)
                st.plotly_chart(fig_surface)
            # After each visualization
            insights = get_visualization_insights(
                viz_type=viz_type,

                feature_x=feature_x,
                feature_y=feature_y,
                dist_type=dist_type if 'dist_type' in locals() else None,
                plot_3d_type=plot_3d_type if 'plot_3d_type' in locals() else None,
                z_feature=z_feature if 'z_feature' in locals() else None
            )
            st.markdown(insights)

        if viz_type == "Radius Plot" or viz_type == "All Plots":
            st.subheader("Radius Plot Analysis")

            selected_features = st.multiselect(
                "Select Features for Radius Plot",
                selected_numeric,
                default=selected_numeric[:5]
            )

            use_hue = st.checkbox("Add categorical comparison")
            if use_hue:
                radius_hue = st.selectbox("Select Category for Comparison",
                                          selected_categorical)

            if len(selected_features) > 1:
                radius_fig = create_radius_plot(data, selected_features,
                                                radius_hue if use_hue else None)
                st.plotly_chart(radius_fig)
            # After each visualization
            insights = get_visualization_insights(
                viz_type=viz_type,

                feature_x=feature_x,
                feature_y=feature_y,
                dist_type=dist_type if 'dist_type' in locals() else None,
                plot_3d_type=plot_3d_type if 'plot_3d_type' in locals() else None,
                z_feature=z_feature if 'z_feature' in locals() else None
            )
            st.markdown(insights)

        # Interactive elements for all plots
        st.sidebar.markdown("### Plot Controls")
        if st.sidebar.checkbox("Show Plot Settings"):
            st.sidebar.slider("Plot Opacity", 0.0, 1.0, 0.7)
            st.sidebar.color_picker("Pick Plot Color", "#1f77b4")

    st.write("### Thank you for using the Advanced Analytics Dashboard!")

    # Advanced Data Cleaning and EDA Tab
    with tab5:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Advanced Data Analysis & Preprocessing")

        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
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

        # 1. Data Quality Analysis
        st.subheader("1. Data Quality Overview")

        # Missing value analysis
        missing_data = data.isnull().sum()
        missing_percent = (missing_data / len(data)) * 100

        quality_df = pd.DataFrame({
            'Missing Values': missing_data,
            'Missing Percentage': missing_percent,
            'Data Type': data.dtypes
        })

        st.write(quality_df)

        # 2. Statistical Analysis
        st.subheader("2. Statistical Analysis")

        # Numeric columns analysis
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

        if st.checkbox("Show Detailed Statistical Analysis"):
            stats_df = data[numeric_cols].agg([
                'mean', 'median', 'std', 'min', 'max',
                lambda x: x.quantile(0.25),
                lambda x: x.quantile(0.75),
                'skew', 'kurtosis'
            ]).round(2)

            stats_df.index = ['Mean', 'Median', 'Std Dev', 'Min', 'Max',
                              '25th Percentile', '75th Percentile',
                              'Skewness', 'Kurtosis']
            st.write(stats_df)

        # 3. Advanced Visualizations
        st.subheader("3. Advanced Visualizations")

        # Visualization 1: Distribution Analysis
        if st.checkbox("Show Distribution Analysis"):
            selected_num_col = st.selectbox("Select Column for Distribution", numeric_cols)

            fig = make_subplots(rows=2, cols=1,
                                subplot_titles=('Distribution Plot', 'Box Plot'))

            # Add histogram
            fig.add_trace(
                go.Histogram(x=data[selected_num_col], name="Distribution"),
                row=1, col=1
            )

            # Add box plot
            fig.add_trace(
                go.Box(x=data[selected_num_col], name="Box Plot"),
                row=2, col=1
            )

            fig.update_layout(height=800, title_text=f"Distribution Analysis of {selected_num_col}")
            st.plotly_chart(fig)

        # Visualization 2: Time Series Analysis
        if st.checkbox("Show Survival Analysis"):
            fig = px.line(data.groupby('Age')['Survival Months'].mean().reset_index(),
                          x='Age', y='Survival Months',
                          title='Average Survival Months by Age')
            st.plotly_chart(fig)

        # Visualization 3: Feature Relationships
        if st.checkbox("Show Feature Relationships"):
            selected_features = st.multiselect("Select Features for Analysis",
                                               numeric_cols,
                                               default=numeric_cols[:3])

            if len(selected_features) > 0:
                correlation_matrix = data[selected_features].corr()

                fig = px.imshow(correlation_matrix,
                                labels=dict(color="Correlation"),
                                x=correlation_matrix.columns,
                                y=correlation_matrix.columns,
                                color_continuous_scale='RdBu')

                st.plotly_chart(fig)

        # Visualization 4: Categorical Analysis
        if st.checkbox("Show Categorical Analysis"):
            categorical_cols = data.select_dtypes(include=['object']).columns
            selected_cat = st.selectbox("Select Categorical Feature", categorical_cols)

            value_counts = data[selected_cat].value_counts().reset_index()
            value_counts.columns = ['Category', 'Count']

            fig = px.pie(value_counts,
                         values='Count',
                         names='Category',
                         title=f'Distribution of {selected_cat}')
            st.plotly_chart(fig)

        # Visualization 5: Bivariate Analysis
        if st.checkbox("Show Bivariate Analysis"):
            num_col = st.selectbox("Select Numeric Feature", numeric_cols, key='bivar_num')
            cat_col = st.selectbox("Select Categorical Feature", categorical_cols, key='bivar_cat')

            fig = px.violin(data, x=cat_col, y=num_col,
                            box=True, points="all",
                            title=f'Distribution of {num_col} across {cat_col}')
            st.plotly_chart(fig)

        # 4. Outlier Detection
        st.subheader("4. Outlier Detection")

        if st.checkbox("Show Outlier Analysis"):
            selected_col = st.selectbox("Select Column for Outlier Detection", numeric_cols)

            Q1 = data[selected_col].quantile(0.25)
            Q3 = data[selected_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[selected_col] < (Q1 - 1.5 * IQR)) |
                            (data[selected_col] > (Q3 + 1.5 * IQR))]

            st.write(f"Number of outliers detected: {len(outliers)}")

            fig = px.box(data, y=selected_col,
                         title=f'Outlier Analysis for {selected_col}')
            st.plotly_chart(fig)

        # Feature Engineering Tab
    with tab6:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Data Processing & Feature Engineering")

        # 1. Feature Creation Section
        st.subheader("1. Feature Creation")

        # Age Groups
        if st.checkbox("Create Age Groups"):
            age_mean = data_actual['Age'].mean()
            age_std = data_actual['Age'].std()

            actual_age = (data_actual['Age'] * age_std) + age_mean

            data['Age_Group'] = pd.cut(data_actual['Age'],
                                       bins=[0, 30, 45, 60, 75, 100],
                                       labels=['Young', 'Middle', 'Senior', 'Elder', 'Advanced'])

            # Visualize age distribution with calculated statistics
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=('Age Distribution', 'Age Groups'))

            # Original age distribution
            fig.add_trace(
                go.Histogram(x=data_actual['Age'], name="Age Distribution"),
                row=1, col=1
            )

            # Age groups distribution
            age_group_counts = data['Age_Group'].value_counts()
            fig.add_trace(
                go.Bar(x=age_group_counts.index, y=age_group_counts.values, name="Age Groups"),
                row=1, col=2
            )

            fig.update_layout(height=400, title_text="Age Analysis")
            st.plotly_chart(fig)

        # Survival Risk Score
        if st.checkbox("Generate Survival Risk Score"):
            data['Risk_Score'] = (
                    data['Tumor Size'] * 0.3 +
                    data['Reginol Node Positive'] * 0.4 +
                    data['Age'] * 0.3
            ).round(2)

            fig = px.histogram(data, x='Risk_Score',
                               title='Distribution of Risk Scores',
                               nbins=30)
            st.plotly_chart(fig)

        # 2. Feature Transformation
        st.subheader("2. Advanced Transformations")

        transform_type = st.selectbox(
            "Select Transformation Method",
            ["Log Transform", "Box-Cox", "Yeo-Johnson", "Quantile"]
        )

        numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
        selected_col = st.selectbox("Select Column for Transformation", numeric_cols)

        if transform_type and selected_col:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=('Original Distribution', 'Transformed Distribution'))

            # Original Distribution
            fig.add_trace(
                go.Histogram(x=data[selected_col], name="Original"),
                row=1, col=1
            )

            # Transform data based on selection
            if transform_type == "Log Transform":
                transformed_data = np.log1p(data[selected_col])
            elif transform_type == "Box-Cox":
                transformed_data = stats.boxcox(data[selected_col] + 1)[0]
            elif transform_type == "Yeo-Johnson":
                transformed_data = stats.yeojohnson(data[selected_col])[0]
            else:  # Quantile
                transformer = QuantileTransformer(output_distribution='normal')
                transformed_data = transformer.fit_transform(data[selected_col].values.reshape(-1, 1)).flatten()

            # Transformed Distribution
            fig.add_trace(
                go.Histogram(x=transformed_data, name="Transformed"),
                row=1, col=2
            )

            fig.update_layout(height=400, title_text=f"{transform_type} Transformation")
            st.plotly_chart(fig)

        # 3. Feature Interactions
        st.subheader("3. Feature Interactions")

        if st.checkbox("Generate Interaction Features"):
            selected_features = st.multiselect(
                "Select 2 Features for Interaction",
                numeric_cols,
                default=numeric_cols[:2]
            )

            if len(selected_features) >= 2:
                # Multiplication Interaction
                data[f'{selected_features[0]}_{selected_features[1]}_interaction'] = (
                        data[selected_features[0]] * data[selected_features[1]]
                )

                # Ratio Interaction
                data[f'{selected_features[0]}_{selected_features[1]}_ratio'] = (
                        data[selected_features[0]] / (data[selected_features[1]] + 1)
                )

                st.write("New Interaction Features:")
                st.write(data[[f'{selected_features[0]}_{selected_features[1]}_interaction',
                               f'{selected_features[0]}_{selected_features[1]}_ratio']].describe())

        # 4. Dimensionality Reduction
        st.subheader("4. Dimensionality Reduction")

        dim_reduction = st.selectbox(
            "Select Dimensionality Reduction Method",
            ["PCA", "t-SNE", "UMAP"]
        )

        if st.checkbox("Apply Dimensionality Reduction"):
            # Prepare numeric data
            X = data[numeric_cols].fillna(0)
            X_scaled = StandardScaler().fit_transform(X)

            if dim_reduction == "PCA":
                # Fit PCA for selected components
                pca_full = PCA()
                pca_full.fit(X_scaled)

                # Create x-axis values as list
                components = list(range(1, len(pca_full.explained_variance_ratio_) + 1))

                # Create scree plot
                fig_scree = px.line(
                    x=components,  # Now using list instead of range
                    y=pca_full.explained_variance_ratio_,
                    title=f'Scree Plot: Explained Variance Ratio for All features',
                    labels={'x': 'Principal Component', 'y': 'Explained Variance Ratio'}
                )
                fig_scree.add_scatter(
                    x=components,  # Now using list instead of range
                    y=np.cumsum(pca_full.explained_variance_ratio_),
                    name='Cumulative Variance Ratio'
                )
                st.plotly_chart(fig_scree)

                # Display variance explained for selected features
                st.write("#### Variance Explained Metrics")
                cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
                for i, var in enumerate(cumulative_var, 1):
                    st.write(f"• {i} components explain {var:.1%} of variance")

                # Component loadings heatmap for selected features
                reducer = PCA(n_components=2)
                reduced_data = reducer.fit_transform(X_scaled)

                loadings = pd.DataFrame(
                    reducer.components_.T,
                    columns=['PC1', 'PC2'],
                    index=numeric_cols
                )
                fig_loadings = px.imshow(
                    loadings,
                    title='PCA Component Loadings for Selected Features',
                    labels=dict(x='Principal Components', y='Features')
                )
                st.plotly_chart(fig_loadings)

                # Final PCA visualization
                fig_final = px.scatter(
                    x=reduced_data[:, 0], y=reduced_data[:, 1],
                    color=data['Status'],
                    title=f'PCA Visualization of Selected Features',
                    labels={'x': 'PC1', 'y': 'PC2'}
                )
                ##st.plotly_chart(fig_final)
            elif dim_reduction == "t-SNE":
                reducer = TSNE(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(StandardScaler().fit_transform(X))
            elif dim_reduction == "UMAP":
                reducer = UMAP(n_components=2, random_state=42)
                reduced_data = reducer.fit_transform(StandardScaler().fit_transform(X))

            # Visualization
            fig = px.scatter(
                x=reduced_data[:, 0], y=reduced_data[:, 1],
                color=data['Status'],
                title=f'{dim_reduction} Visualization'
            )
            st.plotly_chart(fig)


    # Modeling Tab
    with tab7:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        st.header("Model Development & Evaluation")

        # Model Selection Section
        st.subheader("1. Model Selection")
        model_type = st.selectbox(
            "Select Model Type",
            ["Classification", "Clustering", "Regression"]
        )

        if model_type == "Classification":
            # Classification Models
            st.write("### Survival Prediction Models")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(data['Status'])  # This will convert 'Alive'/'Dead' to 0/1

            # Feature Selection
            X = data[['Age', 'Tumor Size', 'Reginol Node Positive']]

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training
            models = {
                'Random Forest': RandomForestClassifier(random_state=42),
                'XGBoost': XGBClassifier(random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42)
            }

            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1': f1_score(y_test, y_pred, average='weighted')
                }

            # Results Visualization
            results_df = pd.DataFrame(results).T
            fig = px.bar(results_df, barmode='group',
                         title='Model Performance Comparison')
            st.plotly_chart(fig)

            # Feature Importance
            if st.checkbox("Show Feature Importance"):
                rf_model = models['Random Forest']
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': rf_model.feature_importances_
                })
                fig = px.bar(importance_df, x='Feature', y='Importance',
                             title='Feature Importance')
                st.plotly_chart(fig)

        elif model_type == "Clustering":
            st.write("### K-Means Clustering Analysis")

            # Feature Selection for Clustering
            numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
            selected_features = st.multiselect(
                "Select Features for Clustering (Choose 2)",
                options=numerical_features,
                default=["Tumor Size", "Age","Reginol Node Examined"]
            )

            if len(selected_features) == 2:
                X_clustering = data[selected_features]

                # Elbow Method
                if st.checkbox("Show Elbow Method"):
                    inertias = []
                    K = range(1, 10)
                    for k in K:
                        kmeans = KMeans(n_clusters=k, random_state=42)
                        kmeans.fit(X_clustering)
                        inertias.append(kmeans.inertia_)

                    fig = px.line(x=K, y=inertias,
                                  title='Elbow Method for Optimal k',
                                  labels={'x': 'k', 'y': 'Inertia'})
                    st.plotly_chart(fig)

                # K-Means Clustering
                n_clusters = st.slider("Select Number of Clusters", 2, 8, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                data['Cluster'] = kmeans.fit_predict(X_clustering)

                fig = px.scatter(data, x=selected_features[0], y=selected_features[1],
                                 color='Cluster',
                                 title='K-Means Clustering Results')
                st.plotly_chart(fig)

                # Silhouette Score
                silhouette_avg = silhouette_score(X_clustering, data['Cluster'])
                st.write(f"Silhouette Score: {silhouette_avg:.3f}")

        else:  # Regression
            st.write("### Survival Months Prediction")

            # Feature and Target Selection
            X = data[['Age', 'Tumor Size', 'Reginol Node Positive']]
            y = data['Survival Months']

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model Training
            reg_models = {
                'Linear': LinearRegression(),
                'Random Forest': RandomForestRegressor(random_state=42),
                'XGBoost': XGBRegressor(random_state=42)
            }

            reg_results = {}
            for name, model in reg_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                reg_results[name] = {
                    'R2': r2_score(y_test, y_pred),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'MSE': mean_squared_error(y_test, y_pred)
                }

            # Results Visualization
            reg_results_df = pd.DataFrame(reg_results).T
            fig = px.bar(reg_results_df, barmode='group',
                         title='Regression Model Performance')
            st.plotly_chart(fig)

            # Learning Curves
            if st.checkbox("Show Learning Curves"):
                selected_model = st.selectbox("Select Model", list(reg_models.keys()))
                train_sizes, train_scores, test_scores = learning_curve(
                    reg_models[selected_model], X, y, cv=5,
                    train_sizes=np.linspace(0.1, 1.0, 10))

                fig = px.line(x=train_sizes,
                              y=[train_scores.mean(axis=1), test_scores.mean(axis=1)],
                              title=f'Learning Curves - {selected_model}',
                              labels={'x': 'Training Examples', 'y': 'Score'})
                st.plotly_chart(fig)

    with tab8:
        st.markdown("""
        # Technical Documentation
        
        ### Data Sources & Attribution

        1. Breast Cancer Wisconsin (Diagnostic) Dataset
            - Source: UCI Machine Learning Repository
            - URL: https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)

        2. Clinical Breast Cancer Data
            - Source: SEER Breast Cancer Data
            - URL: https://www.kaggle.com/datasets/sujithmandala/seer-breast-cancer-data

        Created by: Saraswathi Baskaran
        GitHub: [@saraswathibas](https://github.com/saraswathibas)
        
        ## System Architecture

        ### Core Components
        - **Frontend**: Streamlit-based interactive web interface
        - **Data Processing**: Pandas, NumPy, SciPy
        - **Machine Learning**: Scikit-learn, XGBoost
        - **Visualization**: Plotly, Seaborn, Matplotlib

        ### Module Structure
        ```
        main()
        ├── production_space()
        │   ├── risk_assessment()
        │   ├── survival_prediction()
        │   └── treatment_planning()
        └── data_science_space()
            ├── data_overview()
            ├── search()
            ├── advanced_cleaning()
            ├── visualizations()
            ├── advanced_analysis()
            ├── feature_engineering()
            └── modeling()
        ```

        ## Data Processing Pipeline

        ### Data Preprocessing
        - Standard scaling implementation
        - Missing value handling with KNN imputation
        - Categorical encoding
        - Feature normalization

        ### Feature Engineering
        - Age grouping with custom bins
        - Risk score calculation using weighted features
        - Interaction features generation
        - Dimensionality reduction (PCA, t-SNE, UMAP)

        ## Machine Learning Models

        ### Classification Models
        - Random Forest Classifier
        - XGBoost Classifier
        - Logistic Regression

        ### Regression Models
        - Linear Regression
        - Random Forest Regressor
        - XGBoost Regressor

        ### Clustering
        - K-means clustering with elbow method
        - Silhouette score evaluation

        ## Key Functions

        ### Risk Assessment
        ```python
        def calculate_risk_score(age, tumor_size, nodes_positive, grade):
            '''
            Calculate patient risk score based on clinical parameters

            Parameters:
            - age: int (18-100)
            - tumor_size: float (0.1-200.0 mm)
            - nodes_positive: int (0-50)
            - grade: int (1-3)

            Returns:
            - risk_score: float (0-100)
            '''
        ```

        ### Survival Prediction
        ```python
        def predict_survival(age, stage, treatments):
            '''
            Generate survival predictions

            Parameters:
            - age: int
            - stage: str ('I', 'II', 'III', 'IV')
            - treatments: list[str]

            Returns:
            - dict: {'months': list, 'probability': list}
            '''
        ```

        ## Visualization Components

        ### Interactive Plots
        - Scatter plots with dynamic feature selection
        - Correlation heatmaps
        - Distribution analysis
        - Survival curves
        - Model performance comparisons

        ### Custom Styling
        ```python
        fig.update_layout(
            title_text="Analysis Title",
            xaxis_title="X Label",
            yaxis_title="Y Label",
            plot_bgcolor='white',
            width=800,
            height=500
        )
        ```

        ## Error Handling

        ### Data Validation
        - Required fields validation
        - Value range checks
        - Data type verification

        ### Model Error Handling
        - Cross-validation implementation
        - Model performance monitoring
        - Prediction confidence intervals

        ## Performance Optimization

        ### Techniques
        - Data caching with @st.cache_data
        - Efficient data transformations
        - Streamlined visualization rendering

        ### Memory Management
        - Selective feature loading
        - Batch processing for large datasets
        - Resource cleanup

        ## Dependencies
        - streamlit==1.24.0
        - pandas==1.5.3
        - numpy==1.24.3
        - scikit-learn==1.2.2
        - plotly==5.14.1
        - xgboost==1.7.5
        - seaborn==0.12.2
        - matplotlib==3.7.1

        ## Deployment

        ### Environment Setup
        ```bash
        python -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
        ```

        ### Running the Application
        ```bash
        streamlit run app.py
        ```

        ## Future Enhancements
        1. Additional model architectures
        2. Enhanced feature engineering
        3. Advanced visualization capabilities
        4. Improved error handling
        5. Extended documentation coverage
        """)


def main():
    st.title("Breast Cancer Analysis Platform")

    space = st.sidebar.selectbox(
        "Select Space",
        ["Production Space", "Data Science Space"]
    )

    if space == "Production Space":
        production_space()
    else:
        data_science_space()

if __name__ == "__main__":
    main()
