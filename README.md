# 🩺 Medifraud – Detecting Fraudulent Providers with ML

----

## 🧠 Introduction
Healthcare fraud not only inflates medical costs but also risks patient safety. This project builds an automated system to flag potentially fraudulent providers by analyzing large-scale Medicare claims data. We develop a full ML pipeline—from EDA to model tuning—to deliver actionable fraud detection insights.

---

## 🏥 Healthcare Fraud Overview
Fraud Definition: Billing for services not rendered, falsifying diagnoses, or overcharging Medicare.

Business Impact: Fraudulent claims cost the U.S. healthcare system over $60 billion annually.

Need: Manual auditing is time-consuming and error-prone. ML can reduce audit loads and improve fraud catch rates.

---

## 📦 Dataset Source
Dataset: Medicare Provider Fraud Detection

Files Used:

Inpatient & Outpatient Claims

Beneficiary Info

Labels for Fraud/No-Fraud

Volume: Over 500K+ claim records across ~1,300 providers

---

## 📊 Exploratory Data Analysis (EDA)
>
>
Loaded and merged inpatient, outpatient, and beneficiary training datasets along with fraud labels.

Encoded all categorical variables using LabelEncoder and converted the target to binary (1 = fraud).

Handled missing values using mode or median strategies depending on the column type.

Removed columns with 100% missing values and re-filled partial NaNs robustly.

Used ANOVA F-test (SelectKBest) to select top 30 features most correlated with fraud.

Saved the selected features and labels to a CSV for modeling (Xy_selected_top30.csv).

Analyzed fraud frequency across key categorical variables like Race, Gender, Chronic Conditions.

Engineered a new feature: TotalChronicConditions and explored its relationship to fraud probability.

---

## 🔧 Fraud Data Preprocessing
Merged inpatient, outpatient, beneficiary, and label files

Label encoded all categorical columns

Imputed missing values (mode for categorical, median for numerical)

Applied ANOVA F-test to select top 30 features

Engineered new features: Total chronic conditions, claim duration, fraud density per provider

---

## 🔎 Key Observations and Findings

During the initial exploration of the Medicare dataset, several important patterns emerged, particularly from the beneficiary-level and provider-level data.

### 🧑‍⚕️ Patient-Level Insights

- Certain beneficiaries appeared to be at a higher risk or active target of fraud.
- Patients with high **reimbursement amounts** stood out as potential fraud cases.
- Several patients were observed to have unusually **high deductible payments**.
- A subset of those with high reimbursements or deductibles also had **multiple chronic conditions**, increasing their vulnerability.

### 🏥 Provider-Level Insights

By comparing fraudulent vs. non-fraudulent providers across inpatient and outpatient claims, a few key differences were identified:

- Fraudulent providers tend to show distinct patterns in terms of **claim amounts**, **service volume**, and **procedure diversity**.
- Some features like physician involvement or repeated diagnosis/procedure codes were more frequent among fraudulent claims.

### 🌍 Geographic and Demographic Factors

- Certain **states and counties** appear to have higher concentrations of potentially fraudulent activity.
- A patient’s **age range**, **geographic location**, **total claim amount**, and **primary physician** may collectively signal higher fraud risk.
- These features can provide useful flags for investigators in prioritizing fraud detection efforts.

---

## 📌 Visual Insights and What They Reveal

- **Class Distribution of `PotentialFraud`**
  - 📉 Shows imbalance between fraudulent and non-fraudulent claims—critical for metric selection and resampling strategy.
![Alt Text](files/cdpf.png)

- **Missing Value Matrix**
  - 🔍 Highlights columns with missing data, helping decide between imputation or exclusion.

- **Correlation Heatmap**
  - 🌡️ Identifies multicollinearity between numerical features and reveals strong fraud-related signals.
![Model Diagram](files/ch.png)

- **Pairplot of Top Numerical Features**
  - 📈 Visualizes separation between fraud and non-fraud across top features—useful for feature selection.

- **Barplot of Fraud Rate by State**
  - 🗺️ Shows how fraud probability varies by state, indicating potential geographic patterns.
![Alt Text](files/barplotFD.png)

- **Histograms of Top 5 Numerical Features**
  - 🧮 Helps understand feature distributions, skewness, and outliers that may affect modeling.
![Model Diagram](files/h5nf.png)

- **Countplot of Categorical Features by Fraud**
  - 🧾 Compares how fraud rates vary across categories like Gender, Race, and RenalDiseaseIndicator.
![Model Diagram](files/cpcf.png)

- **Countplot of `TotalChronicConditions`**
  - 💊 Examines how the number of chronic conditions relates to fraud frequency.

- **KDE Plot: `TotalChronicConditions` by Fraud**
  - 📊 Displays conditional density to assess how chronic illness count influences fraud probability.
![Model Diagram](files/kdeplot.png)

---

## 🤖 Modeling & Evaluation

Multiple supervised learning models were trained and evaluated to detect fraudulent Medicare providers. Performance was measured using accuracy, F1-score, precision, recall, and ROC-AUC—especially prioritizing F1-score and ROC-AUC due to class imbalance.

### 📋 Model Performance Comparison

| Model               | Accuracy ↑ | ROC-AUC ↑ | F1-Score ↑ | Precision ↑ | Recall ↑ |
|--------------------|------------|-----------|-------------|-------------|-----------|
| Logistic Regression | 0.89       | 0.74      | 0.46        | 0.52        | 0.42      |
| Random Forest       | 0.92       | 0.85      | 0.59        | 0.58        | 0.61      |
| XGBoost             | 0.93       | 0.88      | 0.63        | 0.61        | 0.64      |
| **LightGBM**        | **0.94**   | **0.91**  | **0.66**    | **0.64**    | **0.68**  |

> ✅ **Best Model:** **LightGBM** was selected due to its best overall performance across all key metrics, especially ROC-AUC and F1-score. It offers high fraud detection capability while minimizing false positives.

---

## 📊 Model Interpretability

Below are the key visualizations used to evaluate, explain, and select the best fraud detection model (LightGBM). Each plot provides specific insights crucial for performance optimization and transparency in healthcare applications.

### ✅ Confusion Matrix – LightGBM
- Shows actual vs. predicted classes (TP, FP, FN, TN) to assess classification quality.
- Helps evaluate how well fraud vs. non-fraud is detected.
- Critical in understanding false positives and false negatives.
![Model Diagram](files/cmlgbm.png)

### 🥇 Model Comparison by F1 Score
- Compares all trained models based on their F1 score.
- LightGBM outperforms others, achieving the best precision-recall balance.
- Justifies LightGBM as the selected final model.
![Model Diagram](files/mcf1s.png)

### 📈 ROC Curve – LightGBM
- Plots True Positive Rate (TPR) vs. False Positive Rate (FPR).
- LightGBM curve is closest to the top-left corner, indicating strong classification.
- Highlights the model’s ability to separate fraud from non-fraud effectively.
![Model Diagram](files/roclightgbm.png)

### 📊 Feature Importance – LightGBM
- Shows the top contributing features in the model based on information gain.
- Variables like `IPAnnualReimbursementAmt` and `ChronicCond_*` rank high.
- Aids in transparency and domain validation of model decisions.
![Model Diagram](files/fi.png)


### 📉 Precision-Recall Curve – LightGBM
- Evaluates model performance under class imbalance (rare fraud cases).
- LightGBM maintains strong precision even at higher recall levels.
- More informative than ROC in imbalanced datasets.
![Model Diagram](files/prc.png)


### 🌊 SHAP Beeswarm Plot – LightGBM (Directional Impact)
- Global explanation of how each feature affects model output across all predictions.
- Red points increase fraud probability; blue points reduce it.
- Explains direction and magnitude of impact for top features.
![Model Diagram](files/shapplot.png)


### 💡 Threshold Tuning Visualization – Optimize Precision/Recall/F1
- Helps decide the best probability threshold (not always 0.5).
- Highlights trade-offs between precision, recall, and F1 score.
- Essential in fraud detection to balance missed fraud vs. over-flagging.
![Model Diagram](files/thresholdprecisionrecall.png)


### 🔎 SHAP Force Plot – Individual Prediction Breakdown
- Explains a single fraud prediction for one provider in detail.
- Shows how specific feature values push the prediction toward fraud or not.
- Useful for case-level investigation and stakeholder trust.
![Model Diagram](files/SHAPForcePlot.png)

---

## 🎯 Threshold Tuning and Final Selection
To determine the optimal threshold for fraud classification, model performance was evaluated at multiple probability cutoffs using precision-recall analysis and fraud prediction distribution plots. An initial threshold of 0.5 flagged 718 out of 1353 providers as potentially fraudulent but leaned toward over-prediction. Visualizations of threshold impact showed that increasing the threshold reduced fraud predictions while improving classification certainty. At approximately 0.42, precision and recall curves intersected, indicating a practical balance between sensitivity and specificity. Based on this tradeoff, a threshold of 0.42 was finalized. The resulting predictions identified 927 providers as fraudulent and 426 as non-fraudulent, offering a more balanced and reliable submission outcome.

---

## 🧾 Conclusion

This project demonstrates a robust machine learning pipeline for detecting healthcare fraud in Medicare claims data by combining domain-driven feature engineering, model optimization, and interpretability techniques.

Key takeaways:

- **LightGBM** emerged as the best-performing model with a strong balance across F1 score, AUC, and precision-recall tradeoffs, making it suitable for high-stakes fraud detection.
- **Advanced feature selection** and rigorous preprocessing significantly improved model generalization while keeping training efficient.
- **Threshold tuning** was critical; a default 0.5 threshold flagged too many providers. Instead, **0.42** was selected based on precision-recall balance.
- **SHAP and interpretability tools** revealed influential features such as total claim amount, chronic conditions, and physician identifiers, enhancing transparency for investigators.
- **EDA insights** suggested that geography, patient chronicity, and claim behavior could signal fraud hotspots, which aligns with real-world fraud patterns in healthcare.

By integrating explainable AI and business-critical metrics, this solution offers a practical, scalable, and trustworthy tool for targeting fraudulent providers in large-scale medical billing systems.
