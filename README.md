# Chronic-Kidney-Disease-Prediction Using ML
This project focuses on building a machine learning pipeline to predict Chronic Kidney Disease (CKD) using patient medical attributes. The workflow includes comprehensive data preprocessing, exploratory data analysis, feature encoding, multiple classification models, and hyperparameter tuning.

Skills Used: Machine Learning | Python | Scikit-learn | GridSearchCV
The following classification models were trained:
1) K-Nearest Neighbors (KNN)
2) Decision Tree Classifier

📌 Project Overview
Chronic Kidney Disease is often underdiagnosed and can lead to severe health consequences if not identified early.
This project builds a classification model capable of predicting CKD using numerical and categorical features extracted from patient health records.

Key Features
✔️ Data Preprocessing: The dataset contains both numeric and categorical features with missing values. We apply a two-step imputation strategy:

1️⃣ Random Sampling Imputation: Used for features with large numbers of missing values to preserve distribution.
def random_value_imputation(feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

2️⃣ Mode Imputation: Used for categorical or low-missing-value columns.
def impute_mode(feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

✔️ Label Encoding: Categorical variables were converted into numerical form using Label Encoding:
    -> Converts categories into 0/1 or integer labels
    -> Essential for ML algorithms that require numeric inputs

📊 Exploratory Data Analysis (EDA): Performed detailed analysis including:
Distribution plots
Boxplots
Missing value analysis
Correlation heatmaps
Detection of patterns between features & CKD outcome
Visualizations help understand data patterns and variable importance.

🤖 Machine Learning Models Used: The following classification models were trained:
K-Nearest Neighbors (KNN)
Decision Tree Classifier

🔧 Hyperparameter Tuning: Used GridSearchCV to find the best model parameters.

Benefits:
Systematic parameter search
Cross-validated scoring
Better optimized model performance

📈 Model Evaluation: Evaluation metrics include:
Accuracy
Precision, Recall, F1-score
Confusion Matrix
Cross-validation score
