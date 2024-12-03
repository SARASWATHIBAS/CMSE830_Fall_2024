# CMSE830_Fall_2024

# SEER Breast Cancer Detection

## Description
This project aims to analyze and visualize breast cancer data to improve understanding of key factors associated with cancer stages and outcomes.


### Key Objectives:
- Data Collection and Preparation: Handling missing values, removing duplicates, and encoding categorical variables.
- Exploratory Data Analysis: Visualizing relationships between various features and their impact on cancer stages.
- Development of a Streamlit app: To present interactive visualizations and analyses.

## Dataset
The primary dataset used for this project is the SEER Breast Cancer Dataset, which includes various features related to breast cancer diagnosis and treatment. 

# Breast Cancer Analysis Application

## Overview
Advanced analytics tool for breast cancer data visualization and prediction using machine learning techniques.

## Features
- Interactive data visualization
- Statistical analysis
- Machine learning models
- Feature engineering
- Real-time predictions

## Data Dictionary
| Feature | Description | Type | Range/Values |
|---------|-------------|------|--------------|
| Age | Patient's age | Numeric | 20-90 |
| Race | Patient's race | Categorical | White, Black, Asian |
| Marital Status | Marital status | Categorical | Single, Married, Divorced |
| T Stage | Tumor stage | Categorical | T1, T2, T3, T4 |
| N Stage | Node stage | Categorical | N0, N1, N2, N3 |
| Grade | Cancer grade | Categorical | 1, 2, 3 |
| Tumor Size | Size in mm | Numeric | 1-150 |
| Status | Survival status | Binary | Alive, Dead |

## Modeling Approach

### Data Preprocessing
1. Missing value imputation using KNN
2. Feature scaling with StandardScaler
3. Categorical encoding using Label Encoding
4. Feature selection based on correlation analysis

### Machine Learning Pipeline
1. Model Selection:
   - Random Forest Classifier
   - XGBoost
   - Logistic Regression

2. Hyperparameter Tuning:
   - Grid Search CV
   - Cross-validation (k=5)

3. Model Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score

## Project Structure
breast-cancer-analysis/ ├── data/ ├── app.py ├── requirements.txt └── README.md


## Installation
```bash
git clone https://github.com/username/breast-cancer-analysis
cd breast-cancer-analysis
pip install -r requirements.txt

Usage
streamlit run app.py

Contributing
Fork the repository
Create feature branch
Commit changes
Push to branch
Create Pull Request

# Description of files:
## Final_Term_Project.ipynb

Jupyter notebook with the full project analysis, including EDA data cleanin and interpretations.

## README.md

Provides an overview of the project, objectives, usage instructions and key findings.

## SEER Breast Cancer Dataset.csv

The dataset used for the analysis containing demographic and clinical data.

## app_final.py

Streamlit app for interactive data analysis and visualizations.

## requirements.txt

Lists all necessary dependencies for running the project.
Include the following essential packages:

streamlit==1.25.0
pandas==2.1.1
numpy==1.24.4
matplotlib==3.8.0
seaborn==0.12.2
scikit-learn==1.3.1
This ensures that anyone using your project can easily set up the environment with the necessary dependencies.

## Setup Instructions

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)


## Installation Instructions
1. Clone the repository:
git clone https://github.com/SARASWATHIBAS/CMSE830_Fall_2024.git
2. Navigate to the project directory:
cd CMSE830_Fall_2024
3. Install the required packages:
pip install -r requirements.txt

Contact
Email: baskar12@msu.edu
LinkedIn: https://www.linkedin.com/in/saraswathibaskaran/

This documentation provides:
1. Clear project overview
2. Detailed data dictionary
3. Comprehensive modeling approach
4. Professional repository structure
5. Installation and usage instructions
6. Contributing guidelines

The repository structure follows industry best practices and makes the project accessible to other developers.
