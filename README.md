# CodeAlpha – Disease Prediction from Medical Data

**CodeAlpha Machine Learning Internship – Task 4**
Author: El Kassiri Douaa

Predicts whether a breast tumour is **malignant or benign** from 30 measurements of cell nuclei (UCI Breast Cancer Wisconsin dataset, 569 patients).

## Approach
1. Load the dataset and explore feature correlations
2. Stratified 80/20 train–test split
3. Standardise features inside a `Pipeline` (no data leakage)
4. Compare **Logistic Regression, SVM, Random Forest, XGBoost** with 5-fold cross-validation
5. Tune the SVM with `GridSearchCV`, optimising **recall** (missing a sick patient is the costliest error)
6. Evaluate with Accuracy, Precision, Recall, F1 and ROC-AUC

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| SVM (RBF) | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 | 0.994 |
| Logistic Regression | 0.965 | 0.975 | 0.929 | 0.951 | 0.996 |
| XGBoost | 0.965 | 1.000 | 0.905 | 0.950 | 0.993 |

Best model: **SVM (RBF), tuned with C=10** → 97% accuracy, malignant precision 1.00, recall 0.93.

All plots (confusion matrix, ROC curves, feature importance, correlation heatmap) are saved in `outputs/`.

## How to run
```bash
pip install -r requirements.txt
python disease_prediction.py
```

## Tech
Python · pandas · scikit-learn · XGBoost · matplotlib · seaborn
