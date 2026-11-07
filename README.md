# Classify, Compare, Deploy

Multi-class classification of dry beans with five machine learning models, compared on one dataset and deployed as a Streamlit app.

## a. Problem statement
Predict the variety of a dry bean (7 classes) from 16 numeric size and shape measurements taken from images of the bean. Five classifiers (Logistic Regression, Decision Tree, KNN, Naive Bayes, Support Vector Machine) are trained on the same data, evaluated with six metrics, and compared. [REWRITE IN YOUR OWN WORDS]

## b. Dataset description
- **Dataset:** Dry Bean Dataset, UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
- **Instances:** 13,611 originally; 13,543 after removing 68 exact duplicate rows (all in the HOROZ class)
- **Features:** 16 numeric (12 dimension features and 4 shape-form features)
- **Target:** `Class`, 7 varieties: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY
- **Missing values:** none
- **Class balance:** imbalanced, from DERMASON (3,546 rows) down to BOMBAY (522 rows)
- **Split:** 80/20 stratified train/test split with a fixed random state. `test_data.csv` is the test part (features and `Class`).
- **Preprocessing:** labels encoded with `LabelEncoder`; features standardised inside the pipelines of Logistic Regression, KNN and SVM.
- **Credit:** Koklu, M. and Özkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture. DOI: https://doi.org/10.24432/C50S4B (CC BY 4.0).

## c. GitHub repository link
https://github.com/RajshreeKhoche/classify-compare-deploy

Live Streamlit app: [PASTE LINK AFTER DEPLOYMENT]

## d. Models used

Metrics are computed on the 20% test split. AUC is one-vs-rest with macro averaging; precision, recall and F1 use weighted averaging.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| Decision Tree | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| KNN | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| Naive Bayes | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |
| Support Vector Machine | [fill] | [fill] | [fill] | [fill] | [fill] | [fill] |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | [write after seeing results] |
| Decision Tree | [write after seeing results] |
| KNN | [write after seeing results] |
| Naive Bayes | [write after seeing results] |
| Support Vector Machine | [write after seeing results] |
| **Overall, Winner for your dataset?** | [model name and why, based on the table] |
