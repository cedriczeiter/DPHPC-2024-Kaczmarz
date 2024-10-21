import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("benchmark_results.csv")
plt.errorbar(data["Dim"], data["AvgTime"], yerr=data["StdDev"], fmt='o')
plt.xlabel("Dim")
plt.ylabel("Average Time (s)")
plt.title("Benchmark of our Algorithm")
plt.show()
