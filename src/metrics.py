from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """
    return {
        "f1_macro":   float(f1_score(y_true, y_pred, average="macro")),
        "f1_class0":  float(f1_score(y_true, y_pred, average=None)[0]),
        "f1_class1":  float(f1_score(y_true, y_pred, average=None)[1]),
        "precision":  float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":     float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy":   float(accuracy_score(y_true, y_pred)),
    }
