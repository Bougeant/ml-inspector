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

# Regression model

```python
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_predict

from ml_inspector import *
```

## Load classification dataset

```python
dataset = fetch_openml(name='california_housing', version=1)
```

```python
dataset.keys()
```

```python
features = dataset["feature_names"]
```

```python
df = pd.DataFrame(data=dataset["data"], columns=features)
df["target"] = dataset["target"]
```

```python
df["longitude"] = df["longitude"].astype(float)
df["latitude"] = df["latitude"].astype(float)
```

```python
df["rooms_per_households"] = df["total_rooms"] / df["households"]
df["bedrooms_per_households"] = df["total_bedrooms"] / df["households"]
df["persons_per_household"] = df["population"] / df["households"]
```

```python
X = df[features]
```

```python
y = df["target"]
```

## Exploratory Data Analysis

```python
plot_regression_features_distribution(df, features=features, target="target")
```

## Train binary classification model

```python
encoder = OneHotEncoder(sparse_output=False)
encoder.set_output(transform="pandas")
```

```python
column_encoder = ColumnTransformer(
    [("one_hot_encoding", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), ["ocean_proximity"])], 
    remainder="passthrough",
    force_int_remainder_cols=False,
)
column_encoder.set_output(transform="pandas")
```

```python
pipeline = Pipeline(
    steps=[
        ("encoding", column_encoder),
        ("missing_value", SimpleImputer(strategy="constant", fill_value=-1)),
        ("random_forest", GradientBoostingRegressor(n_estimators=100, max_depth=5, min_samples_leaf=5)),
    ]
)
```

```python
pipeline.fit(X, y)
```

## Make predictions

```python
y_pred = {
    "Training": pipeline.predict(X),
    "Cross-Validation": cross_val_predict(pipeline, X, y, cv=5)
}
```

```python
from ml_inspector.predictions_regression import plot_regression_predictions
```

```python
plot_regression_predictions(y_true=y, y_pred=y_pred, show_density=True)
```

## Feature importance

```python
plot_feature_importance(pipeline, X, y, scoring="r2", importance_type="removal", max_nb=20)
```

## Learning curve

```python
plot_learning_curves(pipeline, X, y, scoring="r2")
```

## Partial dependence

```python
plot_partial_dependence(pipeline, X)
```

```python

```
