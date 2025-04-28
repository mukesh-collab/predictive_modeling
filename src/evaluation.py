"""
Module for evaluating the trained model.
"""
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model using test data.

    Args:
        model: Trained model.
        X_test (array): Test features.
        y_test (array): Test labels.

    Returns:
        dict: Evaluation metrics.
        ndarray: Confusion matrix.
    """
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    metrics = {
        "accuracy": accuracy,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1_score": report['weighted avg']['f1-score']
    }
    return metrics, conf_matrix
