import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, brier_score_loss, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Custom module import
from main_functions import (evaluate_and_plot, run_lasso_selection, calculate_vif,
                            compare_resampling, calculate_timi, load_and_explore, plot_eda)


# ==========================================
# 1. DATA ACQUISITION & INITIAL EXPLORATION
# ==========================================

data = load_and_explore('pred_data.csv')

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
# Handle placeholders like 'Unknown' and -1 as missing values
data.rename(columns={'Hypothension': 'Hypotension', 'Hyperthension' : 'Hypertension'}, inplace=True)
nonnumeric_mask = pd.to_numeric(data.stack(), errors='coerce').isna().unstack()
negative_mask = (data == -1)
general_mask = nonnumeric_mask | negative_mask

data = data.mask(general_mask, np.nan)
nan_count = data.isna().sum()
print(f'NA %: {sum(nan_count / len(data)) * 100:.2f}%')

# Model-based imputation
iter_imputer = IterativeImputer(max_iter=20, random_state=33)
df_imputed = pd.DataFrame(iter_imputer.fit_transform(data), columns=data.columns)


plot_eda(df_imputed)

# ==========================================
# 3. FEATURE ENGINEERING & BALANCING
# ==========================================
df_imputed["Bmi"] = df_imputed["Weight"] / np.sqrt((df_imputed["Height"] / 100))
df_imputed["TIMI_score"] = df_imputed.apply(calculate_timi, axis=1)
df_imputed["Age_Killip"] = df_imputed['Age'] * df_imputed['Killip_class']
df_imputed["Fragile_Elderly"] = ((df_imputed["Age"] > 70) & (df_imputed["Weight"] < 60)).astype(int)

X = df_imputed.drop('Day30_mortality', axis=1)
y = df_imputed['Day30_mortality']

# Train/Test Split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=33)

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test_raw) # Reference test set

# ==========================================
# 2. DEFINING THE FOUR SCENARIOS
# ==========================================

# 1. Original Dataset (Scaled)
X_train_orig = scaler.fit_transform(X_train_raw)
y_train_orig = y_train

# 2. Oversampled Original (No Lasso)
smote_full = SMOTE(sampling_strategy=0.3, random_state=33)
X_train_over, y_train_over = smote_full.fit_resample(X_train_raw, y_train)
X_train_over_scaled = scaler.fit_transform(X_train_over)

# 3. Resampled Lasso Dataset
lasso_features = [
    'Age', 'Heart_rate', 'Time_To_Relief', 'Previous_myocardial_infarction',
    'Killip_class', 'Weight', 'Anterior_infarct_location', 'Diabetes',
    'Hypotension', 'ST_elevation_leads', 'Gender', 'Smoking',
    'Family_history_of_MI', 'Height'
]

# 4. Undersampled Majority Class
rus = RandomUnderSampler(sampling_strategy=0.8, random_state=33)
X_train_under, y_train_under = rus.fit_resample(X_train_raw, y_train)
X_train_under_scaled = scaler.fit_transform(X_train_under)

# We use undersampled data but filter for Lasso features
X_train_lasso = pd.DataFrame(X_train_under_scaled, columns=X.columns)[lasso_features]
X_test_lasso = pd.DataFrame(X_test_scaled, columns=X.columns)[lasso_features]
y_train_lasso = y_train_under

print("\nVIF Results for Lasso Set:")
print(calculate_vif(X_train_lasso))

# ==========================================
# 6. MODEL SELECTION & TRAINING
# ==========================================
base_models = {
    "Logistic Regression": LogisticRegression(C=5, solver='liblinear',random_state=33, class_weight={0: 1, 1: 15.1},),
    "Random Forest": RandomForestClassifier(n_estimators=614, max_depth=5, criterion='log_loss',
                                             min_samples_leaf=15, random_state=33,
                                            max_features='log2', min_samples_split=30, class_weight={0: 1, 1: 15.1},),
    "SVM": SVC(kernel='rbf', probability=True, shrinking=True,
               random_state=33, C=5),
    "MLP": MLPClassifier(hidden_layer_sizes=(64,32), early_stopping=True, batch_size=32, n_iter_no_change=15,
                         activation='relu', solver='adam', alpha=1e-4,
                         random_state=33, learning_rate='adaptive',
                         learning_rate_init=0.002, validation_fraction=0.1),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=4, splitter='random',
                                            min_samples_split=40, class_weight={0: 1, 1: 15.1},
                                            min_samples_leaf=20, random_state=33),
    "SGD Classifier": SGDClassifier(loss='log_loss', early_stopping=True,
                                    learning_rate='constant', alpha=0.01, random_state=33, penalty='elasticnet',
                                    l1_ratio=0.15),
    "Hist Gradient Booster": HistGradientBoostingClassifier(min_samples_leaf=40, max_iter=2000, early_stopping=True,
                                                            validation_fraction=0.1, n_iter_no_change=15, random_state=33,
                                                            max_depth=10, learning_rate=0.004,
                                                            loss='log_loss'),
    "Bagging Classifier": BaggingClassifier(n_estimators=1000,  random_state=33, n_jobs=-1),
}


scenarios = [
    ("Original", X_train_orig, X_test_scaled, y_train_orig),
    ("Oversampled_Full", X_train_over_scaled, X_test_scaled, y_train_over),
    ("Lasso_Resampled", X_train_lasso, X_test_lasso, y_train_lasso),
    ("Undersampled", X_train_under_scaled, X_test_scaled, y_train_under)
]

results_list = []

for scenario_name, train_x, test_x, train_y in scenarios:
    for model_name, b_model in base_models.items():
        full_name = f"{model_name}_{scenario_name}"

        # Fit Calibrated Model
        model = CalibratedClassifierCV(b_model, method='sigmoid', cv=5)
        model.fit(train_x, train_y)

        probs = model.predict_proba(test_x)[:, 1]

        # F1 Optimization for Threshold
        precision, recall, thresholds = precision_recall_curve(y_test, probs)
        f1_scores = np.divide(2 * (precision * recall), (precision + recall),
                              out=np.zeros_like(precision * recall),
                              where=(precision + recall) != 0)
        best_f1 = np.max(f1_scores) if len(f1_scores) > 0 else 0
        best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        preds = (probs >= best_threshold).astype(int)

        # Assuming evaluate_and_plot returns a dict of metrics
        metrics = evaluate_and_plot(y_test, probs, preds, full_name)
        metrics['F1_Score'] = best_f1
        metrics['Scenario'] = scenario_name
        metrics['Brier'] = brier_score_loss(y_test, probs)
        metrics['Model'] = model_name
        metrics['Best_Threshold'] = best_threshold
        results_list.append(metrics)


df_results = pd.DataFrame(results_list)
print("\nFinal Model Comparison (Sorted by AUC):")
print(df_results.sort_values(by="AUC", ascending=False))