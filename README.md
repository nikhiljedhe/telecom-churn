# Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn in a telecommunications dataset. The goal is to identify customers who are likely to churn (cancel their service) based on their usage patterns and account details. The project includes data exploration, feature engineering, model training and evaluation using various machine learning algorithms, and finally, deploying the best-performing model as a Streamlit web application.

## Dataset
The project uses `telecommunications_Dataset.csv` which contains various features related to customer usage and account information, along with a `churn` column indicating whether a customer churned (1) or not (0).

## Data Analysis and Preprocessing
- **Exploratory Data Analysis (EDA)**: Initial analysis included checking the dataset shape, column information, descriptive statistics, and identifying missing values (none found).
- **Distribution Analysis**: Histograms and box plots were used to visualize the distribution of numerical features and detect outliers.
- **Correlation Analysis**: A heatmap was generated to understand the correlation between numerical features.
- **Feature Engineering**: Several new features were created to enhance the model's predictive power:
    - `total_calls`: Sum of all types of calls (day, evening, night, international).
    - `average_charge_per_call`: Total charge divided by total calls (handling potential division by zero).
    - `customer_lifetime_value`: Total charge multiplied by 2 (as a hypothetical metric).
    - `has_voicemail`: Binary indicator for voice mail plan.
    - `has_international_plan`: Binary indicator for international plan.
- **Class Imbalance Handling**: The dataset showed an imbalance in the `churn` column (2850 non-churn vs. 483 churn). SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the classes in the training data.
- **Data Scaling**: `StandardScaler` was used to scale the features, which is crucial for distance-based algorithms and can improve the performance of others.

## Model Development and Evaluation
The project evaluated several classification models:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**
- **K-Nearest Neighbors (KNN)**

### Model Performance (Accuracy on Test Set):
- Random Forest Classifier: ~98.05%
- Decision Tree Classifier: ~95.05%
- Support Vector Classifier (SVC): ~94.15%
- K-Nearest Neighbors (KNN): ~90.85%
- Logistic Regression: ~85.76%

### Model Optimization
- **Hyperparameter Tuning**: `GridSearchCV` was employed to fine-tune the Random Forest Classifier, as it was the best-performing model initially.
- **Best Parameters**: The optimal hyperparameters found were `{'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'n_estimators': 200}`.
- **Tuned Model Accuracy**: The tuned Random Forest Classifier maintained an accuracy of approximately 98.05% on the test dataset.

## Streamlit Application
A Streamlit web application (`app.py`) has been created to provide an interactive interface for predicting customer churn. 

### How to Run the Streamlit App:
1.  **Install Dependencies**: Ensure you have all necessary libraries installed. A `requirements.txt` file is provided.
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**: Execute the Streamlit application from your terminal or Colab environment:
    ```bash
    streamlit run app.py
    ```

The app will then be accessible via a local URL (e.g., `http://localhost:8501`) or an external URL if running in a cloud environment like Colab.

## Technologies Used
- Python
- Pandas (for data manipulation)
- Scikit-learn (for machine learning models and preprocessing)
- Streamlit (for web application deployment)
- Matplotlib, Seaborn (for data visualization)
- Imbalanced-learn (for SMOTE)
