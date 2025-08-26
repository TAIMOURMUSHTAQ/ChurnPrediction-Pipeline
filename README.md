📋 Project Description
This project implements a complete, production-ready machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. The solution demonstrates best practices for building reusable ML pipelines with scikit-learn's Pipeline API, including data preprocessing, feature engineering, model training, hyperparameter tuning, and model deployment.
The pipeline handles the entire ML workflow from raw data to predictions, making it easy to deploy, maintain, and update. The project emphasizes code reproducibility, modular design, and production readiness.

🚀 Features
•	Complete Data Preprocessing: Automated handling of missing values, categorical encoding, and feature scaling
•	Modular Pipeline Design: Separated preprocessing and modeling steps for maintainability
•	Multiple Algorithm Support: Implementation of both Logistic Regression and Random Forest classifiers
•	Hyperparameter Optimization: Automated tuning using GridSearchCV
•	Model Evaluation: Comprehensive metrics including accuracy, precision, recall, F1-score, and ROC curves
•	Production Export: Complete pipeline serialization using joblib for easy deployment
•	Interactive Web Interface: Streamlit app for real-time predictions

📊 Dataset
The project uses the Telco Customer Churn dataset, which contains information about telecom customers and whether they churned (left the service).
Dataset Features:
•	Demographic info (gender, age, partner, dependents)
•	Account information (tenure, contract type, payment method)
•	Service details (phone service, internet service, online security, etc.)
•	Charges (monthly charges, total charges)
•	Target variable: Churn (Yes/No)

🛠️ Technical Stack
•	Python 3.8+
•	scikit-learn: Pipeline construction, preprocessing, and modeling
•	pandas & numpy: Data manipulation and analysis
•	matplotlib & seaborn: Data visualization
•	joblib: Model serialization and persistence
•	Streamlit: Web application deployment

📁 Project Structure
text
End_to_End_ML_Pipeline/
│
├── data/
│   └── telco_churn.csv          # Raw dataset
│
├── models/
│   └── churn_prediction_pipeline.joblib  # Serialized pipeline
│
├── notebooks/
│   └── Telco_Churn_Analysis.ipynb        # Exploratory data analysis
│
├── src/
│   ├── data_preprocessing.py    # Data cleaning and preparation
│   ├── model_training.py        # Pipeline construction and training
│   └── evaluation.py            # Model evaluation metrics
│
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
🎯 Implementation Details
Data Preprocessing
•	Automated handling of missing values in TotalCharges
•	One-hot encoding for categorical variables
•	Standard scaling for numerical features
•	Target variable transformation (Yes/No to 1/0)
ML Pipeline Architecture
python
# Numerical features pipeline
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical features pipeline  
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combined preprocessing
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Complete pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
Model Training & Optimization
•	Implemented Logistic Regression and Random Forest algorithms
•	Hyperparameter tuning using GridSearchCV
•	Cross-validation for robust performance estimation
•	Best model selection based on ROC AUC score
Evaluation Metrics
•	Accuracy: 80.5%
•	Precision: 67.9%
•	Recall: 53.7%
•	F1-score: 60.0%
•	ROC AUC: 0.85

📈 Results
The optimized Random Forest model achieved:
•	Test Accuracy: 80.5%
•	ROC AUC Score: 0.85
•	Top Features: Contract type, tenure, internet service type, and payment method
The model effectively identifies customers at risk of churning, enabling proactive retention strategies.

🚀 Getting Started
Prerequisites
•	Python 3.8+
•	pip package manager
Installation
1.	Clone the repository:
bash
git clone https://github.com/taimourmushtaq/ChurnPrediction-.git
cd End_to_End_ML_Pipeline
2.	Install dependencies:
bash
pip install -r requirements.txt
3.	Download the dataset from Kaggle and place it in the data/ directory.
Usage
1.	Run the complete pipeline:
bash
python src/model_training.py
2.	Launch the web application:
bash
streamlit run app.py
3.	Access the application at http://localhost:8501

🔧 Customization
To adapt this pipeline for your own dataset:
1.	Modify data_preprocessing.py to handle your data schema
2.	Update the feature lists in model_training.py
3.	Adjust hyperparameters in the GridSearchCV configuration
4.	Customize the Streamlit app interface in app.py


🤝 Contributing
1.	Fork the project
2.	Create your feature branch (git checkout -b feature/AmazingFeature)
3.	Commit your changes (git commit -m 'Add some AmazingFeature')
4.	Push to the branch (git push origin feature/AmazingFeature)
5.	Open a Pull Request


