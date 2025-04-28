"""
Module for result visualization.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_results(metrics, confusion_matrix):
    """
    Plots evaluation metrics and confusion matrix.

    Args:
        metrics (dict): Evaluation metrics.
        confusion_matrix (ndarray): Confusion matrix.
    """
    print("Plotting results...")

    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Metrics bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title("Evaluation Metrics")
    plt.show()
