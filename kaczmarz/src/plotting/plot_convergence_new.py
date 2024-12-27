import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the data
csv_file = "convergence_results.csv"
output_dir = "plots_convergence"  # Directory to save plots
os.makedirs(output_dir, exist_ok=True)

# Read the CSV file
data = pd.read_csv(csv_file)

# Get unique combinations of Problem, Complexity, and Degree
unique_combinations = data[['Problem', 'Complexity', 'Degree']].drop_duplicates()

# Loop through each combination and generate a plot
for _, row in unique_combinations.iterrows():
    problem, complexity, degree = row['Problem'], row['Complexity'], row['Degree']
    
    # Filter data for the current combination
    subset = data[(data['Problem'] == problem) &
                  (data['Complexity'] == complexity) &
                  (data['Degree'] == degree)]
    
    # Sort by precision for consistent plotting
    subset = subset.sort_values(by="Precision")
    
    # Plot with error bars
    plt.figure(figsize=(10, 6))
    plt.errorbar(subset['Precision'], subset['AvgCarpTime'], yerr=subset['StdDevCarpTime'], fmt='o-r', label="Carp")
    plt.errorbar(subset['Precision'], subset['AvgNormalTime'], yerr=subset['StdDevNormalTime'], fmt='o-y', label="Sparse Kaczmarz")
    
    
    # Set logarithmic scale for the x-axis (precision)
    plt.xscale("log")
    plt.yscale('log')

    # Reverse the x-axis
    plt.gca().invert_xaxis()
    
    # Add labels, title, and legend
    plt.xlabel("Precision (log scale)")
    plt.ylabel("Time (ms)")
    plt.title(f"Convergence: Problem {problem}, Complexity {complexity}, Degree {degree}")
    plt.legend()
    
    # Save plot with the required naming convention
    output_file = os.path.join(output_dir, f"convergence_problem_{problem}_complexity_{complexity}_degree_{degree}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Saved plot: {output_file}")