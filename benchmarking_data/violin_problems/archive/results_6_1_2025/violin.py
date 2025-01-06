import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of file paths
file_paths = [
    '../benchmark_results_carp_cg_specific_architecture.csv',
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

    #some renaming:
    if (method == "architecture"):
        method = "GPU Carp-CG"
    elif (method == "cg"):
        method = "Eigen LeastSquaresCG"
    elif (method == "direct"):
        method = "Eigen SparseLU"
    elif (method == "kaczmarz"):
        method = "Kaczmarz"
    elif (method == "cusolver"):
        method = "NVIDIA cuDSS"

    # Iterate over all 8 complexities
    for complexity in range(1, 9):
        # Filter the data to only include rows with the current complexity
        df_filtered = df[df['Complexity'] == complexity]

        # Plotting the violin plot
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Problem', y='Time', data=df_filtered)

        # Customize the plot
        plt.title(f'Violin Plot of Time by Problem (Complexity {complexity}) with {method}')
        plt.xlabel('Problem')
        plt.ylabel('Time (seconds)')

        # Save plot
        plt.savefig(f'violin_plot_{method}_complexity_{complexity}.png')
        plt.close()