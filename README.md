# 🧮 Customer Churn Prediction Project

## 📊 Dataset
The project uses the Bank Customer Churn Prediction dataset, which contains account information for 10,000 customers at a European bank. The dataset includes details such as:
- Credit Score: Customer's credit score
- Balance: Customer’s account balance
- Number of Products: Number of products the customer holds
- Exited: Whether the customer has churned (1) or not (0)

### Dataset Access:
You can download the dataset from Kaggle:
👉 https://www.kaggle.com/datasets/shubhammeshram579/bank-customer-churn-prediction

## 🔍 Overview
This project aims to predict customer churn using machine learning and showcase insights via a Power BI dashboard and a Streamlit web app. The project involves:
- Data preprocessing and transformation
- Handling class imbalance using SMOTE
- ML model training and evaluation
- Model calibration and deployment
- Feature importance extraction
- Data visualization in Power BI

---

## 📂 Project Workflow

### 1. Data Preparation
- Cleaned dataset
- One-hot encoding of categorical variables
- StandardScaler for feature normalization

### 2. Initial Modeling (Before SMOTE)
- Logistic Regression
- Decision Tree
- Random Forest

### 3. Imbalance Handling
- Applied **SMOTE** to handle class imbalance

### 4. Model Evaluation (After SMOTE)
| Model               | Accuracy |
|--------------------|----------|
| XGBoost            | 0.8479   |
| Random Forest      | 0.8430   |
| ANN (MLP)          | 0.7976   |
| SVM                | 0.7933   |
| KNN                | 0.7555   |
| Logistic Regression| 0.7021   |

## 🌲 Final Model Selection

### ✅ Random Forest Classifier
- **Best performer** after evaluation
- **Hyperparameter tuning** using `GridSearchCV`
- **Best Parameters**: 
  - `n_estimators`: 300
  - `min_samples_split`: 2
  - `min_samples_leaf`: 1
  - `max_depth`: None
- **Best F1 Score**: 0.8969
- **Test Accuracy**: 0.8467
- **Model calibration** applied to improve probability scores
- Final **best Random Forest model** saved as `RF.pkl`
- Final **calibrated model** saved as `model.pkl`

## 📁 Repository Contents

- `model.pkl` – ✅ Calibrated Random Forest model  
  📎 [Download](https://drive.google.com/file/d/16e5H5z11LyVmURm4io_PD7tGAXmww1qO/view?usp=drive_link)

- `RF.pkl` – ✅ Best Random Forest model (after hyperparameter tuning)  
  📎 [Download](https://drive.google.com/file/d/10pNZ4BVUEG7XJ2ciGPftZRD5W_hqRPw9/view?usp=drive_link)

- `scaler.pkl` – StandardScaler used for feature scaling

- `Bank_churn_modified.csv` – Modified dataset with two additional columns:
  - `Exit_Probability`: Predicted probability of customer churn
  - `Predicted`: Final predicted status (0 = stay, 1 = churn)

- `feature_importance.csv` – Feature importance scores from the Random Forest model

- `Bank_churn.ipynb` – Jupyter Notebook for:
  - Preprocessing
  - Model training and tuning
  - Model calibration and evaluation

- `app.py` – Source code for the Streamlit application

- `Bak_customer_churn.pbix` – Power BI dashboard file (for advanced data visualization)

- `requirements.txt` – List of required Python packages and versions


## 🚀 Live Applications

- 🔗 **Streamlit App**: [Predict Customer Churn](https://customer-churn-prediction-rkoy.onrender.com/)
- 📊 **Power BI Dashboard**: [Maven Analytics Project](https://mavenanalytics.io/project/31894)

---

## 💡 Key Business Insights

- **Churn Rate**: 20.37% among 10K customers
- **Top Predictors**: Age, account balance, product count
- **High-Risk Segments**: Germany-based and middle-aged customers (45–64 years)
- **Geographic Risk**: Germany has higher churn, possibly due to larger average balances (€119K vs €62K in France/Spain)
- **False Positives**: 244 customers incorrectly flagged — needs model refinement

---

## 🎯 Strategic Recommendations

- 🏦 Retain high-value clients (balance > €200K) using loyalty incentives
- 👥 Personalize engagement for seniors and middle-aged groups
- 🌍 Focused campaigns in Germany to address higher churn risk
- 🧪 Monitor false positives to avoid unnecessary retention efforts

---

## 👩‍💻 Author
[Sudeshna Kundu Mondal](https://github.com/sudeshna1991621)

---

## 📬 Feedback & Contributions
Feel free to raise issues or contribute via pull requests to improve this project.

