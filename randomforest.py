import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, 
                             confusion_matrix, roc_curve, auc, 
                             RocCurveDisplay, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

def preprocessor(X, num, cat):
    transform = ColumnTransformer(
        transformers = [
            ('num', 'passthrough', num),
            ('cat', OneHotEncoder(), cat)
        ]
    )

    return transform.fit_transform(X)

def labelEncode(y):
    le = LabelEncoder()
    return le.fit_transform(y)

def split(X, y, num, cat, test_size=0.2, random_state=42):
    np.random.seed(42)
    X_processed = preprocessor(X, num, cat)
    y_encode = labelEncode(y)
    return X_train, X_test, y_train, y_test = train_test_split(X_processed, y_encode, test_size = test_size, random_state = random_state, stratify=y_encode)
    