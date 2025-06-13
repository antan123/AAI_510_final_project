import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, roc_curve, auc, 
                           ConfusionMatrixDisplay, classification_report, 
                           precision_recall_curve)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

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

def random_forest_classifier(X_train, y_train, **kwargs):
    """Create and train a Random Forest classifier with given parameters."""
    default_params = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'random_state': 42
    }
    # Update default parameters with any provided kwargs
    default_params.update(kwargs)
    
    # Create and train the model
    model = RandomForestClassifier(**default_params)
    return model.fit(X_train, y_train)

def hyperparameter_tuning(X_train, y_train, param_grid=None, n_jobs=-1):
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

def grid_random_forest_classifier(X_train, y_train, **best_params):
    """Create and train a Random Forest classifier with optimized parameters."""
    rf_grid = RandomForestClassifier(**best_params)
    return rf_grid.fit(X_train, y_train)
    
def eval_metrics(y_test, y_pred, y_proba=None, classes=None):
    """Calculate and return various evaluation metrics."""
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

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from trained model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.show()
    return indices, importances