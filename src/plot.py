import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("results.csv")

# Create the plot with error bars and connected lines
plt.errorbar(data["Dim"], data["AvgTime"], yerr=data["StdDev"], fmt='o-', capsize=5, label="Execution Time")

# Set labels and title
plt.xlabel("Dim")
plt.ylabel("Average Time (s)")
plt.title("Benchmark Performance")

# Add grid and legend
plt.grid(True)
plt.legend()

# Save the plot to a file
plt.savefig("benchmark_plot.png")

# Optionally display the plot (if you are running this interactively)
plt.show()
