import pandas as pd
import matplotlib.pyplot as plt


data_normal_dense = pd.read_csv("results_normalsolver_dense.csv")
data_sparse_sparse = pd.read_csv("results_sparsesolver_sparse.csv")
data_random_dense = pd.read_csv("results_randomsolver_dense.csv")

# Create the plot with error bars and connected lines
plt.errorbar(data1["Dim"], data1["AvgTime"], yerr=data1["StdDev"], fmt='o-', capsize=5, label="Normal Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data2["Dim"], data2["AvgTime"], yerr=data2["StdDev"], fmt='o-', capsize=5, label="Sparse Solver Sparse")

# Create the plot with error bars and connected lines
plt.errorbar(data3["Dim"], data3["AvgTime"], yerr=data3["StdDev"], fmt='o-', capsize=5, label="Random Solver Dense")


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
