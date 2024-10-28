import pandas as pd
import matplotlib.pyplot as plt


data_normal_dense = pd.read_csv("results_normalsolver_dense.csv")
data_sparse_sparse = pd.read_csv("results_sparsesolver_sparse.csv")
data_random_dense = pd.read_csv("results_randomsolver_dense.csv")

# Create the plot with error bars and connected lines
plt.errorbar(data_normal_dense["Dim"], data_normal_dense["AvgTime"], yerr=data_normal_dense["StdDev"], fmt='o-', capsize=5, label="Normal Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_sparse_sparse["Dim"], data_sparse_sparse["AvgTime"], yerr=data_sparse_sparse["StdDev"], fmt='o-', capsize=5, label="Sparse Solver Sparse")

# Create the plot with error bars and connected lines
plt.errorbar(data_random_dense["Dim"], data_random_dense["AvgTime"], yerr=data_random_dense["StdDev"], fmt='o-', capsize=5, label="Random Solver Dense")


# Set labels and title
plt.xlabel("Dim")
plt.ylabel("Average Time (s)")
plt.title("Benchmark of our Algorithms")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("benchmark_plot.png")

# Optionally display the plot (if you are running this interactively)
plt.show()
