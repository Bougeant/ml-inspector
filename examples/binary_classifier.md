---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
```

# Binary classification model

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml_inspector import *
```

## Load classification dataset

```python
dataset = load_breast_cancer()
```

```python
X = pd.DataFrame(data=dataset["data"], columns=dataset["feature_names"])
```

```python
y = pd.Series(dataset["target"])
```

```python
class_names = {i: str(c) for i, c in enumerate(dataset["target_names"])}
class_names
```

## Train binary classification model

```python
rf = RandomForestClassifier(n_estimators=20, max_depth=5, min_samples_leaf=5)
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
plot_roc_curves(y, y_prob, class_names, decision_threshold=0.5)
```

## Precision-Recall curves

```python
plot_precision_recall_curves(y, y_prob, class_names, decision_threshold=0.5)
```

## Gain curves

```python
plot_gain_curves(y, y_prob, class_names, decision_threshold=0.5)
```

## Confusion matrix

```python
plot_confusion_matrix(y, y_pred["Cross-Validation"], class_names)
```

## Probability distributions

```python
plot_classification_predictions(y, y_prob["Cross-Validation"], class_names, decision_threshold=0.5, points="all")
```

## Class calibration curves

```python
plot_calibration_curves(y, y_prob["Cross-Validation"], class_names, n_bins=10)
```

## Learning curve

```python
plot_learning_curves(rf, X, y, scoring="roc_auc")
```

## Partial dependence

```python
plot_partial_dependence(rf, X, "worst concavity")
```

```python

```
