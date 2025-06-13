import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             confusion_matrix, roc_curve, auc, 
                             RocCurveDisplay, ConfusionMatrixDisplay, classification_report)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessor(X, num, cat):
    """Preprocess data by applying OneHotEncoding to categorical features and passing through numerical ones."""
    transform = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num),
            ('cat', OneHotEncoder(), cat)
        ]
    )
    return transform.fit_transform(X)

def labelEncode(y):
    """Encode target labels with value between 0 and n_classes-1."""
    le = LabelEncoder()
    return le.fit_transform(y)

def split(X, y, num, cat, test_size=0.2, random_state=42):
    """Split data into train and test sets after preprocessing."""
    np.random.seed(random_state)
    X_processed = preprocessor(X, num, cat)
    y_encode = labelEncode(y)
    return train_test_split(
        X_processed, y_encode, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y_encode
    )

def random_forest_classifier(X_train, y_train, n_jobs=-1, random_state=42):
    """Create and train a Random Forest classifier."""
    rf = RandomForestClassifier(n_jobs=n_jobs, random_state=random_state)
    return rf.fit(X_train, y_train)

def eval_mets(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average = 'weighted')
    class_report = classification_report(y_test, y_pred)
    return accuracy, precision, class_report
    