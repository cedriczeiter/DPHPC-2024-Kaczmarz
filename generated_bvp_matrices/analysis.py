import numpy as np
import matplotlib.pyplot as plt

def plot_sparsity_side_by_side(file_before, file_after):
    # Load data for both matrices
    data_before = np.loadtxt(file_before, dtype=int)
    data_after = np.loadtxt(file_after, dtype=int)
    
    # Separate rows and columns for each matrix
    rows_before, cols_before = data_before[:, 0], data_before[:, 1]
    rows_after, cols_after = data_after[:, 0], data_after[:, 1]
    
    # Create a side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # Plot "Before Reordering"
    axes[0].scatter(cols_before, rows_before, marker='o', color='black', s=1)
    axes[0].invert_yaxis()  # Invert y-axis to match matrix indexing
    axes[0].set_title("Sparsity Pattern Before Reordering")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    # Plot "After Reordering"
    axes[1].scatter(cols_after, rows_after, marker='o', color='black', s=1)
    axes[1].invert_yaxis()  # Invert y-axis to match matrix indexing
    axes[1].set_title("Sparsity Pattern After Reordering")
    axes[1].set_xlabel("Column")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Use the function to plot side by side
plot_sparsity_side_by_side("sparsity_before.txt", "sparsity_after.txt")
