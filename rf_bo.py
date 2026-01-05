import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

"""
This is the file responsible for Bayesian Optimization tuning of the hyperparameters. The objective function        
defines the search space. 
"""

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)

    target = 'Day30_mortality'
    X = df.drop(columns=[target])
    y = df[target]

    X['Hypothension'] = X['Hypothension'].replace('Unknown', np.nan).astype(float)
    X = X.fillna(X.median())

    X = pd.get_dummies(X, drop_first=True)

    return X, y


def objective(trial, X, y):
    params = {
        "criterion": trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"]),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features" : trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "class_weight": "balanced",
        "random_state": 33
    }

    clf = RandomForestClassifier(**params)

    score = cross_val_score(clf, X, y, cv=5, scoring='roc_auc', n_jobs=-1).mean()
    return score


# 3. Run the Optimization
if __name__ == "__main__":
    X, y = load_and_preprocess('pred_data.csv')

    # Create the study with TPESampler (Bayesian)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=33))

    print("Starting Bayesian Optimization...")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=2000)

    print("\n--- Optimization Results ---")
    print(f"Best AUC-Score: {study.best_value:.4f}")
    print("Best Parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")