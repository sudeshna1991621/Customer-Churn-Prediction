# 🧮 Customer Churn Prediction Project

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
| Model              | Accuracy | Precision | Recall  | F1-Score |
|-------------------|----------|-----------|---------|----------|
| Random Forest      | 0.8430   | 0.5940    | 0.6143  | 0.6040   |
| XGBoost            | 0.8479   | 0.6222    | 0.5583  | 0.5885   |
| SVM                | 0.7933   | 0.4797    | 0.7170  | 0.5748   |
| ANN (MLP)          | 0.7976   | 0.4858    | 0.6656  | 0.5617   |
| KNN                | 0.7555   | 0.4238    | 0.7092  | 0.5305   |
| Logistic Regression| 0.7021   | 0.3586    | 0.6703  | 0.4672   |

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
- `churn_model.pkl` – Calibrated Random Forest model
- `modified_dataset.csv` – Dataset with predicted churn probability and classification
- `feature_importance.csv` – Feature importance from Random Forest
- `notebooks/` – Jupyter notebooks for model training, evaluation, and calibration

---

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

