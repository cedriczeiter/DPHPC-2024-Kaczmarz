import pandas as pd
import matplotlib.pyplot as plt


data_sparse_sparse = pd.read_csv("results_sparsesolver_sparse.csv")
data_Eigen_sparse = pd.read_csv("results_eigensolver_sparse.csv")
#data_asynchronous_cpu_sparse = pd.read_csv("results_asynchronous_cpu_sparse.csv")
#data_asynchronous_gpu_sparse = pd.read_csv("results_asynchronous_cuda_sparse.csv")

# Create the plot with error bars and connected lines
plt.errorbar(data_sparse_sparse["Dim"], data_sparse_sparse["AvgTime"], yerr=data_sparse_sparse["StdDev"], fmt='o-', capsize=5, label="Our Solver")

# Create the plot with error bars and connected lines
plt.errorbar(data_Eigen_sparse["Dim"], data_Eigen_sparse["AvgTime"], yerr=data_Eigen_sparse["StdDev"], fmt='o-', capsize=5, label="Eigen Solver")

# Create the plot with error bars and connected lines
#plt.errorbar(data_asynchronous_cpu_sparse["Dim"], data_asynchronous_cpu_sparse["AvgTime"], yerr=data_asynchronous_cpu_sparse["StdDev"], fmt='o-', capsize=5, label="Asynchronous CPU Solver Sparse")

# Create the plot with error bars and connected lines
#plt.errorbar(data_asynchronous_gpu_sparse["Dim"], data_asynchronous_gpu_sparse["AvgTime"], yerr=data_asynchronous_gpu_sparse["StdDev"], fmt='o-', capsize=5, label="Asynchronous GPU Solver Sparse")

# Set labels and title
plt.xlabel("Dim")
plt.ylabel("Average Time (s)")
plt.title("Benchmark of our Sparse Algorithms")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("benchmark_sparse_plot.png")

# Optionally display the plot (if you are running this interactively)
plt.show()
