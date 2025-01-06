import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Increase font size
plt.rcParams.update({'font.size': 16})

# List of file paths
file_paths = [
    '../benchmark_results_carp_cgspecificarchitecture.csv',
    '../benchmark_results_basic_kaczmarz.csv',
    '../benchmark_results_cgmnc.csv',
    '../benchmark_results_cusolverspecificarchitecture.csv',
    '../benchmark_results_eigen_cg.csv',
    '../benchmark_results_eigen_direct.csv',
    '../benchmark_results_eigen_bicgstab.csv'
]

# Method mapping
method_mapping = {
    "cgspecificarchitecture": "GPU iterative CARP-CG",
    "kaczmarz": "CPU iterative Kaczmarz",
    "cgmnc": "CPU iterative CGMNC",
    "cusolverspecificarchitecture": "GPU direct NVIDIA cuDSS",
    "cg": "CPU iterative Eigen CG",
    "direct": "CPU direct Eigen SparseLU",
    "bicgstab": "CPU iterative Eigen BiCGSTAB"
}

# Dictionary to store data for each problem
problem_data = {}

# Iterate over each file
for path in file_paths:
    # Load the data
    df = pd.read_csv(path)

    # Extract the method name from the file path
    method = path.split('_')[-1].split('.')[0]

    # Iterate over each problem
    for problem in df['Problem'].unique():
        # Filter the data to only include rows with the current problem
        df_filtered = df[(df['Problem'] == problem) & (df['Status'] == 'Converged')].copy()

        # Group by dimension and calculate median and confidence interval
        df_grouped = df_filtered.groupby('Dim').agg({'Time': ['median', 'var']}).reset_index()
        df_grouped.columns = ['Dim', 'Time_mean', 'Time_var']
        df_grouped['Method'] = method_mapping.get(method, method)

        # Store the data in the dictionary
        if problem not in problem_data:
            problem_data[problem] = df_grouped
        else:
            problem_data[problem] = pd.concat([problem_data[problem], df_grouped])

# Define a color palette and create a color map
palette = sns.color_palette("bright", len(method_mapping))
color_map = {method: palette[i] for i, method in enumerate(method_mapping.values())}

# Plotting
for problem, data in problem_data.items():
    # Plot for Carp-CG, cuDSS, and SparseLU
    fig, ax1 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Dim', y='Time_mean', hue='Method', style='Method', palette=color_map, data=data[data['Method'].isin(['GPU iterative CARP-CG', 'GPU direct NVIDIA cuDSS', 'CPU direct Eigen SparseLU'])], ax=ax1)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title(f'Running Time vs Dimension for Problem {problem} (CARP-CG against direct solvers)')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Time (seconds)')
    ax1.legend(loc='lower right')
    plt.savefig(f'dimension_vs_time_problem_{problem}_direct_solvers.png', bbox_inches='tight')
    plt.savefig(f'dimension_vs_time_problem_{problem}_direct_solvers.eps', bbox_inches='tight')
    plt.close()

    # Plot for all iterative solvers
    fig, ax2 = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x='Dim', y='Time_mean', hue='Method', style='Method', palette=color_map, data=data[data['Method'].isin(['GPU iterative CARP-CG', 'CPU iterative CGMNC', 'CPU iterative Eigen BiCGSTAB'])], ax=ax2)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title(f'Running Time vs Dimension for Problem {problem} (CARP-CG against iterative solvers)')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Time (seconds)')
    ax2.legend(loc='lower right')
    plt.savefig(f'dimension_vs_time_problem_{problem}_iterative_solvers.png', bbox_inches='tight')
    plt.savefig(f'dimension_vs_time_problem_{problem}_iterative_solvers.eps', bbox_inches='tight')
    plt.close()