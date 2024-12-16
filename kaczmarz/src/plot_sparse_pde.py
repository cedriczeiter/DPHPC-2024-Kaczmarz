import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files and manually assign algorithm names
data_eigeniterative = pd.read_csv("results_eigeniterative_sparse_pde.csv")
data_eigeniterative["Algorithm"] = "Eigen iterative Conjugate Gradient"

data_seqcg = pd.read_csv("results_sparsesolver_sparse_cg_pde.csv")
data_seqcg["Algorithm"] = "Sparse CG"

data_eigeniterative2 = pd.read_csv("results_eigeniterative_2_sparse_pde.csv")
data_eigeniterative2["Algorithm"] = "Eigen iterative BiCGSTAB"

#data_bandedcuda = pd.read_csv("results_banded_cuda_sparse_pde.csv")
#data_bandedcuda["Algorithm"] = "Banded CUDA"

#data_bandedcpu = pd.read_csv("results_banded_cpu_2_threads_sparse_pde.csv")
#data_bandedcpu["Algorithm"] = "Banded CPU"

data_seqnormal = pd.read_csv("results_sparsesolver_sparse_pde.csv")
data_seqnormal["Algorithm"] = "Base algorithm"

data_eigendirect = pd.read_csv("results_eigensolver_sparse_pde.csv")
data_eigendirect["Algorithm"] = "Eigen Direct"

data_cudadirect = pd.read_csv("results_cudadirect_sparse_pde.csv")
data_cudadirect["Algorithm"] = "CUDA Direct"

data_carp = pd.read_csv("results_carp_cuda_sparse_pde.csv")
data_carp["Algorithm"] = "CARP CUDA"

# Combine the datasets into a single DataFrame
data = pd.concat([
    data_eigeniterative,
    data_eigeniterative2,
    #data_bandedcuda,
    #data_bandedcpu,
    data_seqnormal,
    data_eigendirect,
    data_cudadirect,
    data_seqcg,
    data_carp])

# Get unique problems and degrees
problems = data["Problem"].unique()
degrees = data["Degree"].unique()

# Plot results for each problem and degree
for problem in problems:
    for degree in degrees:
        # Filter data for the current problem and degree
        filtered_data = data[(data["Problem"] == problem) & (data["Degree"] == degree)]

        # Increase figure size
        plt.figure(figsize=(12, 6))

        # Plot each algorithm on the same graph
        for algorithm in filtered_data["Algorithm"].unique():
            algorithm_data = filtered_data[filtered_data["Algorithm"] == algorithm]
            plt.errorbar(
                algorithm_data["Dim"],
                algorithm_data["AvgTime"],
                yerr=algorithm_data["StdDev"],
                fmt='o-',
                capsize=5,
                label=algorithm
            )
        
        # Set labels and title
        plt.xlabel("Dimension")
        plt.ylabel("Average Time (s)")
        plt.title(f"Performance for Problem #{problem}, Degree {degree}")

        # Add grid and legend
        plt.grid(True)
        plt.yscale('log')
        plt.xscale('log')
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        # Adjust layout to avoid cropping the legend
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        # Save the plot without cropping
        plt.savefig(f"benchmark_problem_{problem}_degree_{degree}.png", bbox_inches='tight')

        # Show the plot (if interactive)
        plt.show()