import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#increase font size
plt.rcParams.update({'font.size': 14})

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
    "cgspecificarchitecture": "GPU iterative Carp-CG",
    "kaczmarz": "CPU iterative Kaczmarz",
    "cgmnc": "CPU iterative cgmnc",
    "cusolverspecificarchitecture": "GPU direct NVIDIA cuDSS",
    "cg": "CPU iterative Eigen CG",
    "direct": "CPU direct Eigen SparseLU",
    "bicgstab": "CPU iterative Eigen BiCGSTAB"
}

# Selected methods for plotting
selected_methods = ["GPU iterative Carp-CG", "GPU direct NVIDIA cuDSS", "CPU iterative cgmnc", "CPU direct Eigen SparseLU"]

# Iterate over each complexity level
for complexity in range(1, 9):
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), sharey=False)

    # Iterate over each file
    for path in file_paths:


        # Load the data
        df = pd.read_csv(path)




        # Extract the method name from the file path
        method = path.split('_')[-1].split('.')[0]
        method = method_mapping.get(method, method)

        # Check if the method is one of the selected methods
        if method in selected_methods:

            # Filter the data to only include rows with the current complexity and status "Converged"
            df_filtered = df[(df['Complexity'] == complexity) & (df['Status'] == 'Converged')].copy()

            # Calculate the median time for normalization
            median_time = df_filtered['Time'].median()

            # Normalize the 'Time' values by the median
            df_filtered.loc[:, 'NormalizedTime'] = df_filtered['Time'] / median_time

            # Get the index of the subplot
            index = selected_methods.index(method)

            # Plotting the violin plot
            sns.violinplot(x='Problem', y='NormalizedTime', data=df_filtered, ax=axes[index])

            # Customize the plot
            axes[index].set_title(f'{method}')
            axes[index].set_xlabel('Problem')
            axes[index].set_ylabel('Normalized Time by Median')

    # Set the overall title
    fig.suptitle(f'Normalized Violin Plots of Time by Problem (Complexity {complexity})')

    # Adjust layout to reduce whitespace
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure
    plt.savefig(f'normalized_violin_plots_complexity_{complexity}.eps', bbox_inches='tight')
    plt.savefig(f'normalized_violin_plots_complexity_{complexity}.png', bbox_inches='tight')
    plt.close()