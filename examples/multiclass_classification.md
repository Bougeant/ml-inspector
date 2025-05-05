---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

# Multi-class classification model

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import fetch_openml

from ml_inspector import *
```

## Load classification dataset

```python
dataset = fetch_openml(name='iris', version=1)
```

```python
class_names = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
class_names
```

```python
df = dataset["data"]
df["target"] =  dataset["target"].map({v:k for k, v in class_names.items()})
```

```python
df = df.sample(frac=1)
```

```python
X = df.drop(columns=["target"])
```

```python
y = df["target"]
```

## EDA

```python
plot_classification_features_distribution(df, features=X.columns, target="target", class_names=class_names)
```

## Train binary classification model

```python
rf = LogisticRegression()
```

```python
rf.fit(X, y)
```

## Make predictions

```python
y_pred = {
    "Training": rf.predict(X),
    "Cross-Validation": cross_val_predict(rf, X, y, cv=5)
}
```

```python
y_prob = {
    "Training": rf.predict_proba(X),
    "Cross-Validation": cross_val_predict(rf, X, y, cv=5, method="predict_proba")
}
```

## ROC curves

```python
plot_roc_curves(y, y_prob, class_names)
```

## Precision-Recall curves

```python
plot_precision_recall_curves(y, y_prob, class_names)
```

## Gain curves

```python
plot_gain_curves(y, y_prob, class_names)
```

## Confusion matrix

```python
plot_confusion_matrix(y, y_pred["Cross-Validation"], class_names)
```

## Probability distributions

```python
plot_classification_predictions(y, y_prob["Cross-Validation"], class_names, points="all")
```

## Class calibration curves

```python
plot_calibration_curves(y, y_prob["Cross-Validation"], class_names, n_bins=10)
```

## Learning curve

```python
plot_learning_curves(rf, X, y, scoring="roc_auc_ovr", nb_points=10)
```

## Partial dependence

```python
plot_partial_dependence(rf, X, scoring="roc_auc")
```

```python

```
