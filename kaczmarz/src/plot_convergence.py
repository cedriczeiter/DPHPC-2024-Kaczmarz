import pandas as pd
import matplotlib.pyplot as plt


data_normal_dense = pd.read_csv("residuals_normalsolver_dense.csv")
data_sparse_sparse = pd.read_csv("residuals_sparsesolver_sparse.csv")
data_random_dense = pd.read_csv("residuals_randomsolver_dense.csv")

# Create the plot with error bars and connected lines
plt.plot(data_normal_dense["Time"], data_normal_dense["Residual"], 'o-', label="Normal Solver Dense")

# Create the plot with error bars and connected lines
plt.plot(data_sparse_sparse["Time"], data_sparse_sparse["Residual"], 'o-', label="Sparse Solver Sparse")

# Create the plot with error bars and connected lines
plt.plot(data_random_dense["Time"], data_random_dense["Residual"], 'o-', label="Random Solver Dense")


# Set labels and title
plt.xlabel("Time")
plt.ylabel("Residual/Residual(0)")
plt.xscale('log')
plt.yscale('log')
plt.title("Convergence of our Algorithms")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("convergence_plot.png")

# Optionally display the plot (if you are running this interactively)
plt.show()
