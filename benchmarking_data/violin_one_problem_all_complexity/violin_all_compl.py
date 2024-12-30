import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    '../benchmark_results_carp_cg.csv',
    '../benchmark_results_basic_kaczmarz.csv',
    '../benchmark_results_cgmnc.csv',
    '../benchmark_results_cusolver.csv',
    '../benchmark_results_eigen_cg.csv',
    '../benchmark_results_eigen_direct.csv'
]

# Iterate over each file
for path in file_paths:
    # Load the data
    df = pd.read_csv(path)

    # Display the first few rows of the DataFrame to understand its structure
    print(df.head())

    # Extract the method name from the file path
    method = path.split('_')[-1].split('.')[0]

    # Plotting the violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Dim', y='Time', data=df)

    # Customize the plot
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Violin Plot of Time by Dimension with {method} (Log-Log Scale)')
    plt.xlabel('Dimension (log scale)')
    plt.ylabel('Time (seconds, log scale)')

    # Save plot
    plt.savefig(f'violin_plot_{method}_loglog.png')
    plt.close()