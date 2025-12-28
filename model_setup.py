import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score, brier_score_loss, roc_curve, auc
import logging

data = pd.read_csv('pred_data.csv')
print(len(data))

### data exploration: check missing values, check data distribution, correlation matrices, simple tests
print(data.describe())

# Values like 'Unknown' and -1 were found and need to be converted to NA values in order to implement Multivariate Imputation
nonnumeric_mask = pd.to_numeric(data.stack(), errors='coerce').isna().unstack()
negative_mask = (data == -1)
general_mask = nonnumeric_mask | negative_mask


data = data.mask(general_mask, np.nan)
nan_count = data.isna().sum()
print(f'NA %: {sum(nan_count/len(data))*100}.2f')

iter_imputer = IterativeImputer(max_iter=20, random_state=33)

df_imputed = pd.DataFrame(iter_imputer.fit_transform(data), columns=data.columns)

corr_matrix = df_imputed.corr()

### Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation matrix of full dataset")
plt.show()


### Target distribution
sns.countplot(x='Day30_mortality', data=df_imputed)
plt.show()

# !!!!! plot shows that there is a case of class imbalance due to small number of patients who died

# data preprocessing; impute missing values, what is needed for the dataset to become trainable for a model?
X = df_imputed.drop('Day30_mortality', axis=1)
y = df_imputed['Day30_mortality']

# feature selection/engineering based on clinical knowledge

## YET TO ADD

# simple feature engineering try
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=0.95) #PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.2f}")


"""
Since the dataset lacks the exact time-to-event and only contains a binary outcome value, I cannot estimate the "Hazard" (risk over time), 
but I can estimate the Probability of the event occurring within that 30-day window and try out SVMs, logistic regression, decision trees, NNs """

# resampling/upsampling/penalize misclassification if necessary

# model training
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=33)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# probability predictions
y_probs = model.predict_proba(X_test)[:, 1]

# then binary predictions
y_preds = model.predict(X_test)
# model evaluation
print(f"AUC: {roc_auc_score(y_test, y_probs)}")
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_value = roc_auc_score(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression ROC (AUC = {auc_value:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Classifier (AUC = 0.5)')

plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

print(f"F1-score: {f1_score(y_test, y_preds)}")

""" 
this F1 score result demonstrates the class imbalance  
"""

