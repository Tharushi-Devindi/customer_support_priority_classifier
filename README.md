## Project Title
Predictive Model to Classify Customer Support Ticket Priority Levels

## Description
This project develops a machine learning model to automatically classify customer support tickets into predefined priority levels (Low, Medium, High, Critical). The primary objective is to streamline the ticket management process, enabling faster identification and resolution of critical issues, thereby improving customer satisfaction and optimizing resource allocation within a support team. The project covers the entire data science lifecycle, from raw data preprocessing and feature engineering to model training, evaluation, optimization, and explainability.

## Installation
To set up the project and run the Jupyter Notebook, follow these steps:

1.  **Clone the repository (or download the project files):**
    ```bash
    git clone <your-repo-url>
    cd customer_support_priority_classifier
    ```
    (Replace `<your-repo-url>` with the actual URL if you host this on GitHub.)

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to isolate project dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux/Git Bash:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    While your virtual environment is active, install all necessary libraries:
    ```bash
    pip install pandas numpy scikit-learn nltk textblob matplotlib seaborn jupyter shap statsmodels
    ```

4.  **Download the dataset:**
    The dataset is sourced from Kaggle. Download the `customer_support_tickets.csv` file and place it in the `data/raw/` directory within your project structure.
    [https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset]

## Usage
To execute the analysis and observe the model's performance:

1.  **Start Jupyter Notebook:**
    Ensure your virtual environment is active (you should see `(venv)` at the beginning of your terminal prompt).
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:**
    In your web browser, navigate to the `notebooks/` directory and open `customer_ticket_priority_classifier.ipynb`.
3.  **Run Cells:**
    Execute all cells sequentially from top to bottom. The notebook is structured to walk you through each stage of the data science pipeline, with comments and outputs at every step.

## Project Structure
The project adheres to a modular and organized structure, aligning with production-level engineering practices:
Here's a comprehensive `README.md` file generated based on your completed Jupyter Notebook, explaining your process, model choices, and results, while also incorporating the "Production-Level Engineering" mindset for future steps.

-----

````markdown
# Customer Support Ticket Priority Classifier

## Project Title
Predictive Model to Classify Customer Support Ticket Priority Levels

## Description
This project develops a machine learning model to automatically classify customer support tickets into predefined priority levels (Low, Medium, High, Critical). The primary objective is to streamline the ticket management process, enabling faster identification and resolution of critical issues, thereby improving customer satisfaction and optimizing resource allocation within a support team. The project covers the entire data science lifecycle, from raw data preprocessing and feature engineering to model training, evaluation, optimization, and explainability.

## Installation
To set up the project and run the Jupyter Notebook, follow these steps:

1.  **Clone the repository (or download the project files):**
    ```bash
    git clone <your-repo-url>
    cd customer_support_priority_classifier
    ```
    (Replace `<your-repo-url>` with the actual URL if you host this on GitHub.)

2.  **Create and activate a Python virtual environment:**
    It's highly recommended to use a virtual environment to isolate project dependencies.
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux/Git Bash:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    While your virtual environment is active, install all necessary libraries:
    ```bash
    pip install pandas numpy scikit-learn nltk textblob matplotlib seaborn jupyter shap statsmodels
    ```
    *Note: NLTK data (stopwords, wordnet) is downloaded programmatically within the notebook. If you encounter issues, you might need to run `nltk.download('stopwords')` and `nltk.download('wordnet')` in a separate Python script or directly in your activated terminal once.*

4.  **Download the dataset:**
    The dataset is sourced from Kaggle. Download the `customer_support_tickets.csv` file and place it in the `data/raw/` directory within your project structure.
    [https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

## Usage
To execute the analysis and observe the model's performance:

1.  **Start Jupyter Notebook:**
    Ensure your virtual environment is active (you should see `(venv)` at the beginning of your terminal prompt).
    ```bash
    jupyter notebook
    ```
2.  **Open the Notebook:**
    In your web browser, navigate to the `notebooks/` directory and open `customer_ticket_priority_classifier.ipynb`.
3.  **Run Cells:**
    Execute all cells sequentially from top to bottom. The notebook is structured to walk you through each stage of the data science pipeline, with comments and outputs at every step.

## Project Structure
The project adheres to a modular and organized structure, aligning with production-level engineering practices:

````

customer\_support\_priority\_classifier/
├── data/
│   ├── raw/                      \# Contains the original, raw dataset.
│   │   └── customer\_support\_tickets.csv
│   └── cleaned/                  \# Stores the cleaned and preprocessed dataset (output from notebook).
│       └── customer\_support\_tickets\_cleaned.csv
├── notebooks/
│   └── customer\_ticket\_priority\_classifier.ipynb \# The main Jupyter Notebook containing all code for analysis and modeling.
├── models/                       \# (Not explicitly used for saving in this notebook, but good practice for deployment)
├── .env                          \# (Optional) For storing sensitive environment variables (e.g., API keys).
├── .gitignore                    \# Specifies files and directories to be ignored by Git (e.g., virtual environment, data outputs).
└── README.md                     \# This README file, detailing the project.
└── requirements.txt              \# (Optional but recommended) A list of all Python dependencies for easy installation.

````

## Data Source
The dataset utilized in this project is the "Customer Support Ticket Dataset," publicly available on Kaggle. It provides labeled customer support tickets with various attributes suitable for classification tasks.

 **Kaggle Dataset Link:** [https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset)

## Model Details

The development of the predictive model followed a structured pipeline:

### 1. Data Preprocessing & Cleaning

This phase focused on transforming raw data into a clean and suitable format for machine learning.

**Initial Data Overview:** `df.info()` and `df.isnull().sum()` were used to inspect data types, identify non-null counts, and quantify missing values across columns.
* **Irrelevant Column Removal:** Columns with a high proportion of missing values or those deemed non-contributory to the prediction (e.g., identifiers) were removed. Specifically, `Resolution`, `Time to Resolution`, `Customer Satisfaction Rating`, `Ticket ID`, `Customer Name`, and `Customer Email` were dropped.
**Duplicate Row Check:** The dataset was checked for duplicate entries, and none were found.
**Text Cleaning (`Ticket Description`):**
    * Placeholders like `{product_purchase}` and backslashes were removed using regular expressions.
    * All text was converted to lowercase.
    * Punctuation (excluding apostrophes for contractions) and multiple spaces were normalized, and leading/trailing whitespace was trimmed.
    **Note: While NLTK (stopwords, WordNetLemmatizer) was imported, explicit application of stopwords removal and lemmatization to `Ticket Description` after regex cleaning was not performed in the provided notebook cells. This is a key area for future enhancement.**
**Date Feature Extraction:** The `Date of Purchase` column was converted to datetime objects. `purchase_year` and `purchase_month` were extracted to capture temporal patterns. The original `Date of Purchase` column was then dropped. `First Response Time` was also converted to datetime for later feature engineering.
**Categorical Feature Encoding:** Nominal categorical columns (`Ticket Type`, `Ticket Status`, `Ticket Channel`, `Customer Gender`) were transformed using One-Hot Encoding (`pd.get_dummies` with `drop_first=True` to prevent multicollinearity). The resulting boolean columns were converted to integers.
**Target Variable Transformation:** The `Ticket Priority` column, which is the target variable (containing 'Low', 'Medium', 'High', 'Critical'), was ordinally encoded into numerical values (0, 1, 2, 3 respectively) using a predefined mapping.

### 2. Feature Engineering

New features were created to enhance the predictive power of the model.

**Time-based Features:**
    * `response_hour` and `response_dayofweek` were extracted from the `First Response Time` column. Missing values in these new features were imputed using the mode of their respective columns.
    * The original `First Response Time` column was subsequently dropped.
**Text-based Features:**
    * `desc_word_count`: Calculated the total number of words in the cleaned `Ticket Description`.
    * `desc_has_urgent`: A binary (0/1) flag indicating whether the `Ticket Description` contains any of the predefined urgency keywords (`urgent`, `asap`, `immediately`, `emergency`, `now`, `critical`).
    * `desc_sentiment`: The sentiment polarity score of the `Ticket Description` was calculated using `TextBlob`, providing a numerical representation of the emotional tone.
    * *Note: Unlike many text classification tasks, TF-IDF vectorization was not applied to the text fields. The feature engineering focused on simpler, interpretable text attributes.*
**Multicollinearity Assessment:**
    * Variance Inflation Factor (VIF) was used to detect multicollinearity among numerical features. Features with high VIF scores indicate redundancy and can negatively impact model stability and interpretability.
    * `response_dayofweek` (VIF: 43.55) and `desc_word_count` (VIF: 24.76) were identified as highly collinear. These columns were subsequently dropped from the feature set to mitigate multicollinearity.
    * A correlation matrix heatmap was also generated to visually inspect relationships between numerical features.

### 3. Model Training & Optimization

The core of the predictive modeling involved training and fine-tuning classification algorithms.

**Data Splitting:** The preprocessed and engineered dataset was split into training (80%) and testing (20%) sets using `train_test_split`. `stratify=y` was applied to ensure that the distribution of `Ticket Priority` classes was maintained proportionally in both splits, which is crucial for imbalanced datasets.
**Feature Scaling:** `StandardScaler` was fitted on the training data (`X_train`) and then used to transform both `X_train` and `X_test`. This standardizes numerical features to have a mean of 0 and a standard deviation of 1, benefiting models sensitive to feature scales (like Logistic Regression).
**Model Selection:**
    * **Logistic Regression (`LogisticRegression`):** A linear model chosen as a baseline for its simplicity and interpretability. It was trained on the scaled data.
    * **Random Forest Classifier (`RandomForestClassifier`):** An ensemble tree-based model, generally robust to outliers and capable of capturing non-linear relationships. It was trained on the unscaled data (as tree-based models are less sensitive to feature scaling).
**Hyperparameter Tuning (`GridSearchCV`):**
    * `GridSearchCV` with 5-fold cross-validation (`cv=5`) was employed for both models to systematically search for the optimal combination of hyperparameters.
    * The `scoring='f1_weighted'` metric was used for optimization. This metric is particularly suitable for multi-class classification problems, especially when classes might be imbalanced, as it provides a weighted average of the F1-score for each class based on its support.
    * **Logistic Regression Best Parameters:**
        ```
        {'C': 10, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'lbfgs'}
        Best LR score (f1_weighted): 0.254
        ```
    * **Random Forest Best Parameters:**
        ```
        {'class_weight': 'balanced', 'max_depth': 10, 'min_samples_split': 10, 'n_estimators': 100}
        Best RF score (f1_weighted): 0.273
        ```
    * The `class_weight='balanced'` parameter was included in the hyperparameter grids to address the potential class imbalance in the `Ticket Priority` target variable, giving more weight to minority classes during training.

### 4. Model Evaluation

The performance of the optimized models was assessed using standard classification metrics on the unseen test set.

**Evaluation Metrics:** `sklearn.metrics.classification_report` was used to output precision, recall, F1-score, and support for each priority class (0: Low, 1: Medium, 2: High, 3: Critical), along with macro and weighted averages. Overall accuracy was also reported.
**Confusion Matrix:** A confusion matrix was generated and visualized for both Logistic Regression and Random Forest models to provide a detailed breakdown of correct and incorrect classifications per class.

#### **Key Results (on Test Set after Tuning):**

**Logistic Regression:**
    ```
                  precision    recall  f1-score   support

               0       0.24      0.20      0.22       413
               1       0.25      0.33      0.29       438
               2       0.22      0.24      0.23       417
               3       0.24      0.19      0.21       426

        accuracy                           0.24      1694
       macro avg       0.24      0.24      0.24      1694
    weighted avg       0.24      0.24      0.24      1694
    ```
**Random Forest:**
    ```
                  precision    recall  f1-score   support

               0       0.24      0.23      0.24       413
               1       0.25      0.28      0.26       438
               2       0.26      0.26      0.26       417
               3       0.22      0.21      0.21       426

        accuracy                           0.24      1694
       macro avg       0.24      0.24      0.24      1694
    weighted avg       0.24      0.24      0.24      1694
    ```

**Overall Performance Interpretation:**
Both Logistic Regression and Random Forest models achieved an accuracy of approximately 24% and F1-weighted scores of 24%. Given that there are 4 classes, a random guess would yield 25% accuracy. This indicates that the current models are performing no better than random chance. The low precision, recall, and F1-scores across all individual classes further confirm that the models are struggling significantly to learn meaningful patterns from the current feature set to differentiate between ticket priority levels.

### 5. Model Explainability

To understand how the Random Forest model arrived at its predictions, feature importance and SHAP values were analyzed.

**Random Forest Feature Importance:**
    * The `feature_importances_` attribute of the `best_rf_model` was used to identify the most influential features. A bar plot was generated to visualize the top 20 features by their importance scores. This provides a global understanding of which features the model relies on most.
**SHAP (SHapley Additive exPlanations):**
    * `shap.TreeExplainer` was used to compute SHAP values for the Random Forest model's predictions on the test set.
    * A **SHAP bar plot** was generated, showing the mean absolute SHAP value for each feature, which is another robust measure of global feature importance.
    * A **SHAP beeswarm plot** was created. This plot visualizes the distribution of SHAP values for each feature across the dataset, revealing how each feature (e.g., its high or low values) influences the model's output for individual predictions (e.g., pushing the prediction towards a higher or lower priority).

