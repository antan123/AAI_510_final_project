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
    """Perform hyperparameter tuning using GridSearchCV and return the trained GridSearchCV object."""
    if param_grid is None:
        param_grid = {
            'bootstrap': [True, False],
            'max_depth': [20, 50, 100],
            'min_samples_split': [5, 10, 15],
            'n_estimators': [200, 300, 400]
        }
    
    rf = RandomForestClassifier(random_state=42)
    grid_cv = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        verbose=1,
        n_jobs=n_jobs
    )
    grid_cv.fit(X_train, y_train)
    return grid_cv  # Return the entire GridSearchCV object

def grid_rfc(X_train, y_train, **best_params):
    """Create and train a Random Forest classifier with optimized parameters."""
    rf_grid = RandomForestClassifier(**best_params)
    return rf_grid.fit(X_train, y_train)

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
    
def random_xgb(X_train, y_train, **best_params):
    """Create and train a XGBoost with optimized parameters."""
    xgb_random = XGBClassifier(**best_params)
    return xgb_random.fit(X_train, y_train)
    
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

def random_svc(X_train, y_train, **best_params):
    """Create and train a SVC with optimized parameters."""
    svc_random = SVC(**best_params)
    return svc_random.fit(X_train, y_train)

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

import matplotlib.pyplot as plt
import numpy as np

# Model Comparison
def plot_model_comparison(models_dict, X_test, y_test):
    """
    Creates a bar chart comparing performance metrics across multiple models.
    Handles both regular models and GridSearchCV/RandomizedSearchCV objects.
    
    Parameters:
        models_dict (dict): Dictionary of {'model_name': model_object}
        X_test: Test features
        y_test: True labels for test set
    """
    # Metrics to compare
    metrics = {
        'Accuracy': accuracy_score,
        'Precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'Recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
    }
    
    # Prepare results
    results = {}
    model_names = []
    
    for name, model in models_dict.items():
        # Handle GridSearchCV/RandomizedSearchCV objects
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        # Calculate metrics
        y_pred = model.predict(X_test)
        model_metrics = {
            metric_name: metric_fn(y_test, y_pred)
            for metric_name, metric_fn in metrics.items()
        }
        
        results[name] = model_metrics
        model_names.append(name)
    
    # Plot setup
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.2
    index = np.arange(len(metrics))
    
    # Create bars for each model
    colors = plt.cm.rainbow(np.linspace(0, 1, len(model_names)))
    for i, model_name in enumerate(model_names):
        values = [results[model_name][metric] for metric in metrics]
        ax.bar(index + i*bar_width, values, bar_width, 
               label=model_name, color=colors[i])
    
    # Formatting
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, pad=20)
    ax.set_xticks(index + bar_width*(len(model_names)-1)/2)
    ax.set_xticklabels(metrics.keys())
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, model_name in enumerate(model_names):
        for j, metric in enumerate(metrics):
            value = results[model_name][metric]
            ax.text(index[j] + i*bar_width, value + 0.02, f'{value:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

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