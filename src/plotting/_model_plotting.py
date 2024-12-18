import matplotlib.pyplot as plt
def plot_predicted_vs_actual(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

def plot_multi_model_lorenz_curve(y_true, model_predictions, model_labels, title='Lorenz Curve for Model Predictions'):
    """
    Plot the Lorenz curves for multiple models and include Gini index in the legend.
    
    Args:
    y_true (array-like): The actual values.
    model_predictions (list of array-like): A list containing arrays of predictions from different models.
    model_labels (list of str): Labels for each model, to be used in the legend.
    
    """
    y_true = np.array(y_true)  # Ensure y_true is a numpy array
    
    # Create the line of perfect equality
    perfect_line = np.linspace(0, 1, len(y_true))
    plt.figure(figsize=(10, 6))
    plt.plot(perfect_line, perfect_line, 'k--', label='Random baseline')

    for predictions, label in zip(model_predictions, model_labels):
        predictions = np.array(predictions)  # Ensure predictions are numpy arrays
        
        # Sort indices by prediction values
        sorted_indices = np.argsort(predictions)
        sorted_true = y_true[sorted_indices]
        sorted_pred = predictions[sorted_indices]

        # Calculate cumulative sums
        cum_actual = np.cumsum(sorted_true) / sum(sorted_true)
        cum_pred = np.cumsum(sorted_pred) / sum(sorted_pred)

        # Calculate the Gini index
        area_under_lorenz = np.trapz(cum_pred, perfect_line)
        area_under_line = 0.5  # Area under the line of perfect equality
        gini_index = 2 * (area_under_line - area_under_lorenz)

        # Plotting
        plt.plot(perfect_line, cum_pred, label=f'{label} (Gini index: {gini_index:.3f})')

    plt.title(title)
    plt.xlabel('Fraction of observations (ordered by model from lowest to highest outcome)')
    plt.ylabel('Fraction of total outcome')
    plt.legend(title="Models")
    plt.grid(True)
    plt.show()
