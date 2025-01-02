import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# List of file paths
file_paths = [
    '../benchmark_results_carp_cg_specific_architecture.csv',
    '../benchmark_results_basic_kaczmarz.csv',
    '../benchmark_results_cgmnc.csv',
    '../benchmark_results_cusolver.csv',
    '../benchmark_results_eigen_cg.csv',
    '../benchmark_results_eigen_direct.csv'
]

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
        df_filtered = df[df['Problem'] == problem]

        # Group by dimension and calculate mean and variance of time
        df_grouped = df_filtered.groupby('Dim').agg({'Time': ['median', 'var']}).reset_index()
        df_grouped.columns = ['Dim', 'Time_mean', 'Time_var']
        df_grouped['Method'] = method
        
        #some renaming:
        if (method == "architecture"):
            df_grouped['Method'] = "GPU Carp-CG"
        elif (method == "cg"):
            df_grouped['Method'] = "Eigen LeastSquaresCG"
        elif (method == "direct"):
            df_grouped['Method'] = "Eigen SparseLU"
        elif (method == "kaczmarz"):
            df_grouped["Method"] = "Kaczmarz"
        elif (method == "cusolver"):
            df_grouped["Method"] = "NVIDIA cuDSS"

        # Store the data in the dictionary
        if problem not in problem_data:
            problem_data[problem] = df_grouped
        else:
            problem_data[problem] = pd.concat([problem_data[problem], df_grouped])

# Plotting
for problem, data in problem_data.items():
    plt.figure(figsize=(12, 6))

    sns.scatterplot(x='Dim', y='Time_mean', hue='Method', data=data)

    # Add trendlines
    for method in data['Method'].unique():
        method_data = data[data['Method'] == method]
        '''z = np.polyfit(np.log(method_data['Dim']), np.log(method_data['Time_mean']), 1)
        p = np.poly1d(z)
        plt.plot(method_data['Dim'], np.exp(p(np.log(method_data['Dim']))), linestyle='--', label=f'{method} Trendline')'''

    # Customize the plot
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'Running Time vs Dimension for Problem {problem}')
    plt.xlabel('Dimension')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Save plot
    plt.savefig(f'dimension_vs_time_problem_{problem}.png', bbox_inches='tight')
    plt.close()