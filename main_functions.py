from scipy.stats import logistic
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
import os
import seaborn as sns

def load_and_explore(file_path):
    data = pd.read_csv(file_path)
    print(f"Dataset Length: {len(data)}")
    print(data.describe())
    return data

def plot_eda(df):
    # Correlation Matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title("Correlation matrix of full dataset")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # target distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Day30_mortality', data=df)
    plt.title("Target variable Distribution (Day30_mortality)")
    plt.show()


def calculate_and_plot_roc(y_true, y_probs, model_name):
    print(f"AUC: {roc_auc_score(y_true, y_probs)}")
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc_value = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'{model_name} ROC (AUC = {auc_value:.3f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label=f'{model_name} (AUC = 0.5)')

    plt.title(f'ROC Curve - {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.show()


def evaluate_and_plot(y_true, y_probs, y_preds, model_name, scenario_name, set_type):
    results_dir = 'model_results'
    os.makedirs(results_dir, exist_ok=True)

    auc_val = roc_auc_score(y_true, y_probs)
    brier_val = brier_score_loss(y_true, y_probs)
    f1_val = f1_score(y_true, y_preds)

    # 1. Generate and Save ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='red', lw=1, linestyle='--', label='Random (AUC = 0.5)')
    plt.title(f'ROC Curve: {model_name}\n({scenario_name} - {set_type})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_{scenario_name}_{set_type}_roc.png'))
    plt.close()

    # 2. Generate and Save Calibration Plot
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10, strategy='quantile')
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='s', lw=2, label=f'{model_name}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Perfectly Calibrated', color='navy')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual)')
    plt.title(f'Calibration Plot: {model_name}\n({scenario_name} - {set_type})')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'{model_name}_{scenario_name}_{set_type}_calibration.png'))
    plt.close()

    return {
        'Model': model_name,
        'Scenario': scenario_name,
        'Dataset': set_type,
        'AUC': auc_val,
        'F1_Score': f1_val,
        'Brier_Score': brier_val
    }


def calculate_timi(row):
    score = 0
    # Age criteria
    if row['Age'] >= 75:
        score += 3
    elif row['Age'] >= 65:
        score += 2

    # History criteria (DM, HTN, or Angina)
    if (row['Diabetes'] == 1) or (row['Hypertension'] == 1) or (row['Previous_angina_pectoris'] == 1):
        score += 1

    # Hemodynamic criteria
    if row['Heart_rate'] > 100: score += 2
    if row['Hypotension'] == 1: score += 3  # Proxy for SBP < 100

    # Severity criteria
    if row['Killip_class'] > 1: score += 2  # Killip II-IV

    # Physical/Presentation
    if row['Weight'] < 67: score += 1
    if row['Anterior_infarct_location'] == 1: score += 1
    if row['Time_To_Relief'] > 4: score += 1

    return score

def run_lasso_selection(df):
    feat = df.drop('Day30_mortality', axis=1)
    feat = pd.get_dummies(feat, columns=['Killip_class', 'Smoking'], drop_first=True)
    X_lasso = StandardScaler().fit_transform(feat)
    y_lasso = df['Day30_mortality'].values

    lasso_cv = LogisticRegressionCV(cv=5,
                                    l1_ratios=(1,),
                                    use_legacy_attributes=False,
                                    solver='liblinear',
                                    scoring='roc_auc',
                                    random_state=33,
                                    max_iter=10000)
    lasso_cv.fit(X_lasso, y_lasso)

    coef_df = pd.DataFrame({'Feature': feat.columns, 'Coefficient': lasso_cv.coef_[0]})
    coef_df['Abs_Impact'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Impact', ascending=False)
    print("Features selected by Lasso:")
    print(coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False))
    return coef_df


def calculate_vif(X_df):
        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_df.columns
        # Calculating VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(len(X_df.columns))]
        return vif_data.sort_values(by="VIF", ascending=False)

