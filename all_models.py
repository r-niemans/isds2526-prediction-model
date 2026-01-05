import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import  precision_recall_curve

from main_functions import (evaluate_and_plot, run_lasso_selection, calculate_vif,
                            calculate_timi, load_and_explore, plot_eda)


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

df_imputed["Bmi"] = df_imputed["Weight"] / ((df_imputed["Height"] / 100) ** 2)
df_imputed["TIMI_score"] = df_imputed.apply(calculate_timi, axis=1)
df_imputed["Age_Killip"] = df_imputed['Age'] * df_imputed['Killip_class']
df_imputed["Fragile_Elderly"] = ((df_imputed["Age"] > 70) & (df_imputed["Weight"] < 60)).astype(int)

X = df_imputed.drop('Day30_mortality', axis=1)
y = df_imputed['Day30_mortality']

lasso_df = run_lasso_selection(df_imputed)

lasso_features = [
    'Age', 'Heart_rate', 'Time_To_Relief', 'Previous_myocardial_infarction',
    'Killip_class', 'Weight', 'Anterior_infarct_location', 'Diabetes',
    'Hypotension', 'ST_elevation_leads', 'Gender', 'Smoking',
    'Family_history_of_MI', 'Height'
]


X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.20, random_state=33)

print("\nVIF Results for original Set:")
print(calculate_vif(X_train_raw))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

lasso_train = pd.DataFrame(X_train_scaled, columns=X.columns)[lasso_features]
lasso_test = pd.DataFrame(X_test_scaled, columns=X.columns)[lasso_features]

scenarios = {
    "Original" : (X_train_scaled, X_test_scaled),
    "Lasso" : (lasso_train, lasso_test),
}

# ==========================================
# 6. MODEL SELECTION & TRAINING
# ==========================================

base_models = {
    "Logistic Regression": LogisticRegression(C=0.05, solver='liblinear',random_state=33, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=712, max_depth=7, criterion='entropy',
                                             min_samples_leaf=10, random_state=33, bootstrap=True,
                                            max_features='sqrt', min_samples_split=20, class_weight='balanced'),
    "SVM": SVC(kernel='linear', probability=True, shrinking=True,
               random_state=33, C=0.3),
    "MLP": MLPClassifier(hidden_layer_sizes=(16,8), early_stopping=True, batch_size=16, n_iter_no_change=15,
                         activation='logistic', solver='adam', alpha=0.01,
                         random_state=33, learning_rate='adaptive',
                         learning_rate_init=0.01, validation_fraction=0.1),
    "Decision Tree": DecisionTreeClassifier(criterion='entropy', max_depth=4, splitter='random',
                                            min_samples_split=40, class_weight='balanced',
                                            min_samples_leaf=20, random_state=33),
    "SGD Classifier": SGDClassifier(loss='log_loss', early_stopping=True, class_weight='balanced',
                                    learning_rate='constant', alpha=0.01, random_state=33, penalty='elasticnet',
                                    l1_ratio=0.15),
    "Hist Gradient Booster": HistGradientBoostingClassifier(min_samples_leaf=40, max_iter=2000, early_stopping=True,
                                                            validation_fraction=0.1, n_iter_no_change=15, random_state=33,
                                                            max_depth=10, learning_rate=0.004,
                                                            loss='log_loss'),
    "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.01, max_depth=5, random_state=33, verbose=-1),
    "Bagging Classifier": BaggingClassifier(n_estimators=1000,  random_state=33, n_jobs=-1),
}


results_list = []

for scenario_name, (train_x, test_x) in scenarios.items():
    for model_name, b_model in base_models.items():
        model = CalibratedClassifierCV(b_model, method='isotonic', cv=5)
        model.fit(train_x, y_train)

        probs_test = model.predict_proba(test_x)[:, 1]
        prec, rec, thresh = precision_recall_curve(y_test, probs_test)
        f1_s = np.divide(2*(prec*rec), (prec+rec), where=(prec+rec)!=0)
        best_threshold_test = thresh[np.argmax(f1_s)] if len(thresh) > 0 else 0.5
        preds_test = (probs_test >= best_threshold_test).astype(int)


        probs_train = model.predict_proba(train_x)[:, 1]
        prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_train, probs_train)
        f1_s_tr = np.divide(2*(prec_tr*rec_tr), (prec_tr+rec_tr), where=(prec_tr+rec_tr)!=0)
        best_threshold_train = thresh_tr[np.argmax(f1_s_tr)] if len(thresh_tr) > 0 else 0.5
        preds_train = (probs_train >= best_threshold_train).astype(int)

        for set_type, probs_val, target_val, preds_val in [
            ('Test', probs_test, y_test, preds_test),
            ('Train', probs_train, y_train, preds_train)
        ]:
            metrics = evaluate_and_plot(
                y_true=target_val,
                y_probs=probs_val,
                y_preds=preds_val,
                model_name=model_name,
                scenario_name=scenario_name,
                set_type=set_type
            )
            results_list.append(metrics)

df_results = pd.DataFrame(results_list)
print(df_results[['Model', 'Scenario', 'Dataset', 'AUC', 'F1_Score', 'Brier_Score']].sort_values(by="AUC", ascending=False))
df_results.to_csv('model_results.csv', index=False)