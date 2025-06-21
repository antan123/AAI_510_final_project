import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, roc_curve, auc, 
                           ConfusionMatrixDisplay, classification_report, 
                           precision_recall_curve)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path
import shap  

# Preprocessing
def preprocessor(X, num, cat):
    """Preprocess data by standardizing numerical features and OneHotEncoding categorical features."""
    transform = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat)
        ]
    )
    return transform

def labelEncode(y):
    """Encode target labels with value between 0 and n_classes-1."""
    le = LabelEncoder()
    return le.fit_transform(y), le

def split(X, y, num, cat, test_size=0.2, random_state=42):
    """Split data into train and test sets after preprocessing."""
    np.random.seed(random_state)
    preprocessor_obj = preprocessor(X, num, cat)
    X_processed = preprocessor_obj.fit_transform(X)
    y_encode, label_encoder = labelEncode(y)
    return train_test_split(
        X_processed, y_encode, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encode
    ), preprocessor_obj, label_encoder

# Random Forest
def random_forest_classifier(X_train, y_train, **kwargs):
    """Create and train a Random Forest classifier with given parameters."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': 42
    }
    default_params.update(kwargs)
    model = RandomForestClassifier(**default_params)
    return model.fit(X_train, y_train)

def hyperparameter_tuning_rf(X_train, y_train, param_grid=None, n_jobs=-1):
    """Hyperparameter tuning for Random Forest."""
    if param_grid is None:
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [20, 50, 100],
            'min_samples_split': [5, 10, 15],
            'n_estimators': [200, 300, 400]
        }
    rf = RandomForestClassifier(random_state=42)
    grid_cv = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, verbose=1, n_jobs=n_jobs)
    grid_cv.fit(X_train, y_train)
    return grid_cv

# XGBoost
def xgboost_classifier(X_train, y_train, **kwargs):
    """Train an XGBoost classifier with default or custom parameters."""
    default_params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42,
        'eval_metric': 'logloss'  # For binary classification
    }
    default_params.update(kwargs)
    model = XGBClassifier(**default_params)
    return model.fit(X_train, y_train)

def hyperparameter_tuning_xgb(X_train, y_train, param_grid=None, n_jobs=-1):
    """Hyperparameter tuning for XGBoost using RandomizedSearchCV (faster than GridSearch)."""
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
    xgb_model = XGBClassifier(random_state=42)
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        verbose=1,
        n_jobs=n_jobs
    )
    search.fit(X_train, y_train)
    return search

# SVC
def svc_classifier(X_train, y_train, **kwargs):
    """Train a Support Vector Classifier with default or custom parameters."""
    default_params = {
        'C': 1.0,
        'kernel': 'rbf',
        'gamma': 'scale',
        'random_state': 42
    }
    default_params.update(kwargs)
    model = SVC(**default_params, probability=True)  # Enable probability for ROC curves
    return model.fit(X_train, y_train)

def hyperparameter_tuning_svc(X_train, y_train, param_grid=None, n_jobs=-1):
    """Hyperparameter tuning for SVC."""
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    svc = SVC(random_state=42, probability=True)
    grid_cv = GridSearchCV(estimator=svc, param_grid=param_grid, cv=5, verbose=1, n_jobs=n_jobs)
    grid_cv.fit(X_train, y_train)
    return grid_cv

# Shared Evaluation & Plotting
def eval_metrics(y_test, y_pred, y_proba=None, classes=None):
    """Calculate evaluation metrics (unchanged)."""
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    if y_proba is not None and classes is not None and len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
        precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
        metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
    return metrics

def plot_feature_importance(model, feature_names, top_n=20, model_type='rf'):
    """Plot feature importance for any model (RF/XGBoost)."""
    if model_type == 'rf':
        importances = model.feature_importances_
    elif model_type == 'xgb':
        importances = model.feature_importances_
    else:
        raise ValueError("model_type must be 'rf' or 'xgb'")
    
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances ({model_type.upper()})')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()
    return indices, importances

def plot_shap_summary(model, X, feature_names):
    """SHAP summary plot (works for RF/XGBoost)."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, feature_names=feature_names)

# Save/Load Pipeline
def save_pipeline(model, preprocessor, label_encoder, folder_path='model'):
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f'{folder_path}/model.joblib')
    joblib.dump(preprocessor, f'{folder_path}/preprocessor.joblib')
    joblib.dump(label_encoder, f'{folder_path}/label_encoder.joblib')
    print(f"Pipeline saved to {folder_path}")

def load_pipeline(folder_path='model'):
    model = joblib.load(f'{folder_path}/model.joblib')
    preprocessor = joblib.load(f'{folder_path}/preprocessor.joblib')
    label_encoder = joblib.load(f'{folder_path}/label_encoder.joblib')
    print("Pipeline loaded successfully")
    return model, preprocessor, label_encoder