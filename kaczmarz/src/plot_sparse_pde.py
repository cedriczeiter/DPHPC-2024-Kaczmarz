import pandas as pd
import matplotlib.pyplot as plt

# Function to load and plot data for each problem
def plot_results(file_paths, output_plot):
    for file_path in file_paths:
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Extract problem and complexity information
        df['Problem'] = df['File'].str.extract(r'problem(\d+)_complexity')[0].astype(int)
        df['Complexity'] = df['File'].str.extract(r'complexity(\d+).txt')[0].astype(int)
        
        # Group by problem for plotting
        problems = df['Problem'].unique()
        for problem in problems:
            subset = df[df['Problem'] == problem]
            
            plt.errorbar(
                subset['Complexity'], 
                subset['AvgTime'], 
                yerr=subset['StdDev'], 
                label=file_path.split("_")[1],  # Legend label from file name
                marker='o'
            )
        
        plt.title(f"Performance for Problem {problem}")
        plt.xlabel("Complexity")
        plt.ylabel("Average Time (ms)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{output_plot}_problem_{problem}.png")
        plt.clf()

# Specify file paths
file_paths = [
    #"results_banded_cuda_sparse_pde.csv",
    "results_asynchronous_cuda_sparse_pde.csv",
    "results_sparsesolver_sparse_pde.csv",
    "results_eigensolver_sparse_pde.csv",
    "results_eigeniterative_sparse_pde.csv"
]

plot_results(file_paths, "benchmark_plot_pde")