import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files and manually assign algorithm names
data_eigeniterative = pd.read_csv("results_eigeniterative_sparse_pde.csv")
data_eigeniterative["Algorithm"] = "Eigen iterative"

#data_bandedcuda = pd.read_csv("results_banded_cuda_sparse_pde.csv")
#data_bandedcuda["Algorithm"] = "Banded CUDA"

#data_bandedcpu = pd.read_csv("results_banded_cpu_2_threads_sparse_pde.csv")
#data_bandedcpu["Algorithm"] = "Banded CPU"

#data_seqnormal = pd.read_csv("results_sparsesolver_sparse_pde.csv")
#data_seqnormal["Algorithm"] = "Base algorithm"

#data_eigendirect = pd.read_csv("results_eigensolver_sparse_pde.csv")
#data_eigendirect["Algorithm"] = "Eigen Direct"

#data_cudadirect = pd.read_csv("results_cudadirect_sparse_pde.csv")
#data_cudadirect["Algorithm"] = "CUDA Direct"

data_carp = pd.read_csv("results_carp_cuda_sparse_pde.csv")
data_carp["Algorithm"] = "CARP CUDA"

# Combine the datasets into a single DataFrame
data = pd.concat([
    data_eigeniterative,
    #data_bandedcuda,
    #data_bandedcpu,
    #data_seqnormal,
    #data_eigendirect,
    #data_cudadirect,
    data_carp])

# Get the unique problems to iterate through
problems = data["Problem"].unique()

# Plot results for each problem
for problem in problems:
    # Filter data for the current problem
    problem_data = data[data["Problem"] == problem]

    # Increase figure size
    plt.figure(figsize=(12, 6))

    # Plot each algorithm on the same graph
    for algorithm in problem_data["Algorithm"].unique():
        algorithm_data = problem_data[problem_data["Algorithm"] == algorithm]
        plt.errorbar(
            algorithm_data["Complexity"],
            algorithm_data["AvgTime"],
            yerr=algorithm_data["StdDev"],
            fmt='o-',
            capsize=5,
            label=algorithm
        )
    
    # Set labels and title
    plt.xlabel("Complexity")
    plt.ylabel("Average Time (s)")
    plt.title(f"Performance for Problem #{problem}")

    # Add grid and legend
    plt.grid(True)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    # Adjust layout to avoid cropping the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Save the plot without cropping
    plt.savefig(f"benchmark_problem_{str(problem).replace(' ', '_')}.png", bbox_inches='tight')

    # Show the plot (if interactive)
    plt.show()