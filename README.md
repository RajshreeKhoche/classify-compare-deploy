# Classify, Compare, Deploy

Multi-class classification of dry beans with five machine learning models, compared on one dataset and deployed as an interactive Streamlit app.

## a. Problem statement
Predict the variety of a dry bean (7 classes) from 16 numeric size and shape measurements extracted from images of the bean. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes and Support Vector Machine) are trained on the same data and evaluated with six metrics (Accuracy, AUC, Precision, Recall, F1 and MCC), and the results are compared to find the best model. The trained models are served through a Streamlit web app. 

## b. Dataset description
- **Dataset:** Dry Bean Dataset, UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Background:** images of 13,611 bean grains from 7 registered varieties were taken with a high-resolution camera; segmentation and feature extraction produced 16 features per grain (12 dimension features and 4 shape-form features).
- **Instances:** 13,611 originally. 68 exact duplicate rows (all in the HOROZ class) were removed, leaving **13,543** rows.
- **Features:** 16, all numeric (Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1 to ShapeFactor4). The column `AspectRation` keeps the spelling used in the data file.
- **Target:** `Class`, with 7 varieties: DERMASON, SIRA, SEKER, HOROZ, CALI, BARBUNYA, BOMBAY.
- **Missing values:** none.
- **Class balance (after cleaning):** imbalanced, from DERMASON (3,546 rows, 26.2%) down to BOMBAY (522 rows, 3.9%).
- **Split:** 80/20 stratified train/test split with a fixed random state (7): 10,834 training rows and 2,709 test rows. `test_data.csv` is the test part (16 features plus `Class`).
- **Preprocessing:** labels encoded with `LabelEncoder`; features standardised with `StandardScaler` inside the pipelines of Logistic Regression, KNN and SVM (fitted on training data only).
- **Credit:** Koklu, M. and Özkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. Computers and Electronics in Agriculture. DOI: https://doi.org/10.24432/C50S4B (CC BY 4.0).

## c. GitHub repository link
https://github.com/RajshreeKhoche/classify-compare-deploy

**Live Streamlit app:** https://bean-model-comparison-app.streamlit.app/

Repository contents: `app.py` (Streamlit app), `bean_model_training.ipynb` (preprocessing, training and evaluation), `requirements.txt`, `test_data.csv` (test split used in the experiments), `data/Dry_Bean_Dataset.csv` (original data) and `model/` (five saved model pipelines and the label encoder, as `.joblib` files).

## d. Models used

All metrics are computed on the 2,709-row test split. With 7 classes, AUC is one-vs-rest with macro averaging, and precision, recall and F1 use weighted averaging. With weighted averaging, recall is mathematically equal to accuracy, which is why those two columns match.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9232 | 0.9945 | 0.9233 | 0.9232 | 0.9232 | 0.9072 |
| Decision Tree | 0.9033 | 0.9491 | 0.9031 | 0.9033 | 0.9031 | 0.8831 |
| KNN | 0.9214 | 0.9855 | 0.9215 | 0.9214 | 0.9214 | 0.9049 |
| Naive Bayes | 0.7619 | 0.9665 | 0.7614 | 0.7619 | 0.7599 | 0.7125 |
| Support Vector Machine | 0.9288 | 0.9956 | 0.9290 | 0.9288 | 0.9288 | 0.9138 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong for a linear model: accuracy 0.9232 and AUC 0.9945, second best overall. This suggests the bean classes are largely separable using the scaled shape features. |
| Decision Tree | Accuracy 0.9033 and MCC 0.8831, clearly below the top three, and the lowest AUC (0.9491). A single tree gives coarse class probabilities, which likely lowers its AUC. |
| KNN | Accuracy 0.9214 and MCC 0.9049, very close to Logistic Regression. It works well on standardised features but was not the best on any metric. |
| Naive Bayes | The weakest model: accuracy 0.7619 and MCC 0.7125. Its AUC (0.9665) is high, so it ranks the classes reasonably but picks the wrong class more often. Many features (area, perimeter, axis lengths) are strongly related, which likely breaks the independence assumption of Naive Bayes. |
| Support Vector Machine | Best on all six metrics: accuracy 0.9288, AUC 0.9956, F1 0.9288 and MCC 0.9138. |
| **Overall, Winner for your dataset?** | **Support Vector Machine**, the highest on every metric. Logistic Regression is a close second (about 0.6 percentage points lower accuracy), so the gap is small on this single split. |