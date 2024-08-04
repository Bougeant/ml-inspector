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

# Binary classification model

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from ml_inspector import roc_curve
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

## Train binary classification model

```python
rf = RandomForestClassifier(n_estimators=20, max_depth=5, min_samples_leaf=5)
```

```python
rf.fit(X, y)
```

## Make predictions

```python
y_prob = {
    "Training": rf.predict_proba(X), 
    "Cross-Validation": cross_val_predict(rf, X, y, cv=5, method="predict_proba")
}
```

## Plot ROC curves

```python
roc_curve.plot_roc_curves(y, y_prob, class_names=dataset["target_names"], decision_threshold={"Cross-Validation": 0.5})
```

```python
##
```
