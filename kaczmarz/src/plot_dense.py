import pandas as pd
import matplotlib.pyplot as plt


data_normal_dense = pd.read_csv("results_normalsolver_dense.csv")
data_random_dense = pd.read_csv("results_randomsolver_dense.csv")
data_Eigen_dense = pd.read_csv("results_eigensolver_dense.csv")
data_cuda_dense = pd.read_csv("results_cudasolver_dense.csv")
data_asynchronous_cpu_dense = pd.read_csv("results_asynchronous_cpu_dense.csv")
data_asynchronous_gpu_dense = pd.read_csv("results_asynchronous_cuda_dense.csv")


# Create the plot with error bars and connected lines
plt.errorbar(data_normal_dense["Dim"], data_normal_dense["AvgTime"], yerr=data_normal_dense["StdDev"], fmt='o-', capsize=5, label="Normal Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_random_dense["Dim"], data_random_dense["AvgTime"], yerr=data_random_dense["StdDev"], fmt='o-', capsize=5, label="Random Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_Eigen_dense["Dim"], data_Eigen_dense["AvgTime"], yerr=data_Eigen_dense["StdDev"], fmt='o-', capsize=5, label="Eigen Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_cuda_dense["Dim"], data_cuda_dense["AvgTime"], yerr=data_cuda_dense["StdDev"], fmt='o-', capsize=5, label="Cuda Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_asynchronous_cpu_dense["Dim"], data_asynchronous_cpu_dense["AvgTime"], yerr=data_asynchronous_cpu_dense["StdDev"], fmt='o-', capsize=5, label="Asynchronous CPU Solver Dense")

# Create the plot with error bars and connected lines
plt.errorbar(data_asynchronous_gpu_dense["Dim"], data_asynchronous_gpu_dense["AvgTime"], yerr=data_asynchronous_gpu_dense["StdDev"], fmt='o-', capsize=5, label="Asynchronous GPU Solver Dense")

# Set labels and title
plt.xlabel("Dim")
plt.ylabel("Average Time (s)")
#plt.yscale('log')
plt.title("Benchmark of our Dense Algorithms")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("benchmark_dense_plot.png")

# Optionally display the plot (if you are running this interactively)
plt.show()
