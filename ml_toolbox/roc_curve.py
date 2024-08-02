""" ROC curves for classification models. """

import logging

import numpy as np
import plotly.graph_objs as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

logger = logging.getLogger(__name__).setLevel(logging.INFO)


def plot_roc_curves(y_true, y_prob, class_names=None, decision_threshold=None):
    """Plots the ROC curves for a binary or multi-class classification model.

    :param array y_true:
        An array containing the true outcomes.
    :param dict y_prob:
        A dictionary containing the predicted probablities for each class, together
        with their labels. For example:
            {"Train": array([[0.3, 0.7], ...]), "Test": array([[0.4, 0.6], ...])}
        In case of binary classification, only the probablities
        for the positive class may be provided. For example:
            {"Model 1": array([0.1, 0.4]), "Model 2": array([0.2, 0.3])}
    :param dict class_names:
        A dictionary containing the name to display for each class. For example:
            {0: "Class 0", 1: "Class 1", ...}).
    :param dict decision_threshold:
        A dictionary containing the threshold indicating where the boolean decision is
        made from probablity predictions (for binary classification models) together
        with their labels. For example:
            {"Train": 0.5, "Test": 0.7}

    :returns plotly.graph_objs.Figure:
        The figure containing the ROC curves.
    """
    decision_threshold = {} if decision_threshold is None else decision_threshold
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise ValueError("ROC curves are not defined for less than two classes")
    roc_curves = calculate_roc_curves(y_true, y_prob, classes)
    if class_names is None:
        class_names = {c: str(c) for c in classes}
    fig = create_roc_curves_plot(roc_curves, class_names, decision_threshold)
    return fig


def calculate_roc_curves(y_true, y_prob, classes):
    """Returns the ROC curves for both each set of predicted probablities.
    In case of multi-class model, a One vs Rest approach is used to generate ROC
    curves and the micro average ROC curve is added.

    :param array y_true:
        An array containing the true outcomes.
    :param dict y_prob:
        A dictionary containing the predicted probablities for each class, together
        with their labels. For example:
            {"Train": array([[0.3, 0.7], ...]), "Test": array([[0.4, 0.6], ...])}
        In case of binary classification, only the probablities
        for the positive class may be provided. For example:
            {"Model 1": array([0.1, 0.4]), "Model 2": array([0.2, 0.3])}
    :param list classes:
        The class indices.

    :returns dict:
        A dictionary containing the ROC curves for each class for both
        training data and cross-validated predictions.
    """
    roc_curves = {}
    for k, values in y_prob.items():
        if len(classes) == 2:
            roc_curves[k] = get_binary_roc_curve(y_true, values)
        else:
            roc_curves[k] = get_multiclass_roc_curve(y_true, values, classes)
    return roc_curves


def get_binary_roc_curve(y_true, y_prob):
    """Calculates the ROC curve for binary classification models.

    :param array y_true:
        An array containing the true outcomes.
    :param array y_prob:
        An array containing the predicted probablities for the positive class.

    :returns dict:
        A dictionary containing the ROC curve for the positive class.
    """
    roc_curves = {}
    if y_prob.shape[1] == 2:
        y_prob = y_prob[:, 1]
    roc_curves[1] = roc_curve(y_true, y_prob)
    return roc_curves


def get_multiclass_roc_curve(y_true, y_prob, classes):
    """Calculates the ROC curve for multiclass classification models.

    :param array y_true:
        An array containing the true outcomes.
    :param array y_prob:
        An array containing the predicted probablities for each class.
    :param list classes:
        The class indices.

    :returns tuple:
        A dictionary containing the ROC curve for the each class as well as the
        micro-averaged ROC curve.
    """
    roc_curves = {}
    for i, c in enumerate(classes):
        roc_curves[c] = roc_curve(y_true == c, y_prob[:, i])
    roc_curves["Average"] = roc_curve(
        label_binarize(y_true, classes=classes).ravel(), y_prob.ravel()
    )
    return roc_curves


def create_roc_curves_plot(roc_curves, class_names, decision_threshold):
    """Generates a plotly figure of the ROC curves.

    :param dict roc_curves:
        A dictionary containing the ROC curves to display.
    :param dict class_names:
        A dictionary containing the name to display for each class
        (e.g. {1: "Class 1", 2: "Class 2", ...}).
    :param dict decision_threshold:
        A dictionary containing the threshold indicating where the boolean decision is
        made from probablity predictions (for binary classification models) together
        with their labels. For example:
            {"Train": 0.5, "Test": 0.7}

    :returns plotly.graph_objs.Figure:
        The figure containing the ROC curves.
    """
    data = create_roc_curves_plot_data(roc_curves, class_names, decision_threshold)
    layout = create_roc_curves_plot_layout()
    fig = go.Figure(data=data, layout=layout)
    return fig


def create_roc_curves_plot_data(roc_curves, class_names, decision_threshold):
    """Creates the data for the plotly ROC curves plot.

    :param dict roc_curves:
        A dictionary containing the ROC curves to plot.
    :param dict class_names:
        A dictionary containing the name to display for each class
        (e.g. {1: "Class 1", 2: "Class 2", ...}).
    :param dict decision_threshold:
        A dictionary containing the threshold indicating where the boolean decision is
        made from probablity predictions (for binary classification models) together
        with their labels. For example:
            {"Train": 0.5, "Test": 0.7}

    :returns list:
        A list of plotly traces containing the ROC curves.
    """
    class_names = class_names.copy()
    if len(class_names) > 2:
        class_names["Average"] = "Micro-average ROC Curve"
    lines = ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
    data = []
    for j, type_curves in enumerate(roc_curves):
        for i, c in enumerate(roc_curves[type_curves]):
            fpr, tpr, thresholds = roc_curves[type_curves][c]
            score = auc(fpr, tpr)
            name = f"{class_names[c]} ({type_curves}): AUC={score:.2f}"
            line = {"color": DEFAULT_PLOTLY_COLORS[i], "dash": lines[j % len(lines)]}
            data.append(
                go.Scatter(x=fpr, y=tpr, line=line, name=name, legendgroup=str(c))
            )
            if type_curves in decision_threshold:
                threshold = decision_threshold[type_curves]
                data.extend(add_decision_threshold(fpr, tpr, thresholds, threshold))
    data.append(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            line={"color": "black", "dash": "dash"},
            name="Random decision: AUC=0.50",
            mode="lines",
        )
    )
    return data


def add_decision_threshold(fpr, tpr, thresholds, decision_threshold):
    """Adds a decision threshold to the ROC curve for binary classification models.

    :param array fpr:
        An array containing the false positive rate as a function of threshold.
    :param array tpr:
        An array containing the true positive rate as a function of threshold.
    :param array thresholds:
        An array containing the threshold values for the roc curve.
    :param float decision_threshold:
        The selected decision threshold for the model.

    :returns list:
        A list of plotly traces containing the decision threshold.
    """
    index = len([i for i in thresholds if i > decision_threshold])
    threshold_plots = [
        go.Scatter(
            x=[fpr[index - 1], fpr[index - 1]],
            y=[tpr[index - 1], 0],
            line={"color": "black", "width": 2, "dash": "dot"},
            showlegend=True,
            name="Decision threshold",
            legendgroup="Decision threshold",
        ),
        go.Scatter(
            x=[fpr[index - 1], 0],
            y=[tpr[index - 1], tpr[index - 1]],
            line={"color": "black", "width": 2, "dash": "dot"},
            showlegend=False,
            legendgroup="Decision threshold",
        ),
        go.Scatter(
            x=[fpr[index - 1], 0.01],
            y=[0.01, tpr[index - 1]],
            text=[f"{fpr[index - 1]:.2f}", f"{tpr[index - 1]:.2f}"],
            mode="text",
            showlegend=False,
            legendgroup="Decision threshold",
            textposition="top right",
            textfont={"color": "black"},
        ),
    ]
    return threshold_plots


def create_roc_curves_plot_layout():
    """Creates the layout for the ROC curves plot.

    :returns plotly.graph_objs.Layout:
        The layout for the ROC curves plot.
    """
    layout = go.Layout(
        legend={"traceorder": "grouped", "orientation": "h", "y": -0.1},
        width=800,
        height=800,
        yaxis=go.layout.YAxis(title_text="True positive rate", range=(0, 1)),
        xaxis=go.layout.XAxis(title_text="False positive rate", range=(0, 1)),
    )
    return layout
