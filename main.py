"""
Main entry point for predictive modeling using scikit-learn.
"""
import os
from src.data_preprocessing import preprocess_data
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.visualization import plot_results

def main():
    """
    Orchestrates the predictive modeling workflow.
    """
    print("Starting predictive modeling...")

    # Load and preprocess data
    data_path = os.path.join("data", "titanic.csv")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    metrics, confusion_matrix = evaluate_model(model, X_test, y_test)
    print("Evaluation Metrics:", metrics)

    # Visualize results
    plot_results(metrics, confusion_matrix)

if __name__ == "__main__":
    main()


