import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 12})

# Get all files out of the folder data
files = glob.glob('data/*.csv')

# Initialize a dictionary to hold the data
data = {}

# Read each file and store the data
for file in files:
    # Extract the information from the filename
    filename = file.split('/')[-1].split('-')[-1].split('.')[0]
    problem = filename.split('_')[0][-1]
    complexity = filename.split('_')[1][-1]
    degree = filename.split('_')[2][-1]
    key = f'Problem {problem}, Complexity {complexity}, Degree {degree}'
    # Read the data and store it in the dictionary according to filename
    df = pd.read_csv(file)
    # Filter out rows with negative values -> out of iterations
    df = df[(df['Relaxation'] >= 0) & (df['Carp_steps'] >= 0)]
    #filter out rows with relaxation > 0.7 and relaxation < 0.2
    df = df[(df['Relaxation'] <= 0.7) & (df['Relaxation'] >= 0.2)]
    #filter out rows with carp_steps > 40000
    df = df[(df['Carp_steps'] <= 40000)]

    data[key] = df

print("Nr of data points: ", len(data))

# Sort the keys by degree, complexity, and problem
sorted_keys = sorted(data.keys(), key=lambda x: (x.split(', ')[2][-1], x.split(', ')[1][-1], x.split(', ')[0][-1]))

# Define markers for each complexity
complexity_markers = {
    '1': 'o',
    '2': 's',
    '3': '^',
    '4': 'D',
    '5': 'v',
    '6': 'p'
}

# Create a colormap
colormap = cm.get_cmap('tab10')
degree_colors = {}

plt.figure(figsize=(10, 6))

for key, df in data.items():
    degree = key.split(', ')[2][-1]
    complexity = key.split(', ')[1][-1]
    if degree not in degree_colors:
        degree_colors[degree] = colormap(len(degree_colors) / 10.0)
    plt.plot(df['Relaxation'], df['Carp_steps'], marker=complexity_markers[complexity], linestyle='--', label=key, color=degree_colors[degree], linewidth=1)  # Set linewidth to 1

     # Highlight the minima
    if not df.empty:
        min_index = df['Carp_steps'].idxmin()
    else:
        print(f"No valid data points in {file}")
        continue
    min_relaxation = df.loc[min_index, 'Relaxation']
    min_carp_steps = df.loc[min_index, 'Carp_steps']
    plt.plot(min_relaxation, min_carp_steps,  marker=complexity_markers[complexity],  color='red', markersize=10, label=f'Min {key}')


plt.xlabel('Relaxation-Parameter')
plt.ylabel('Iterations')
plt.xlim(0.1352, 0.71)
plt.title('Iterations vs Relaxation-Parameter')
plt.grid(True)

x_coord = 0.18
x_coord_number = 0.145
interval_hor_size = 0.005
# Draw an interval for problem 1
interval_x = [x_coord, x_coord]
interval_y = [0, 2000]
plt.plot(interval_x, interval_y, color='red', lw=2)
plt.plot([interval_x[0] - interval_hor_size, interval_x[0] + interval_hor_size], [interval_y[0], interval_y[0]], color='red', lw=2)
plt.plot([interval_x[1] - interval_hor_size, interval_x[1] + interval_hor_size], [interval_y[1], interval_y[1]], color='red', lw=2)
# Add a big red three to the plot
plt.text(x_coord_number, 0, 'P1', fontsize=20, color='red')

# Draw an interval for problem 2
interval_x = [x_coord, x_coord]
interval_y = [3500, 6500]
plt.plot(interval_x, interval_y, color='red', lw=2)
plt.plot([interval_x[0] - interval_hor_size, interval_x[0] + interval_hor_size], [interval_y[0], interval_y[0]], color='red', lw=2)
plt.plot([interval_x[1] - interval_hor_size, interval_x[1] + interval_hor_size], [interval_y[1], interval_y[1]], color='red', lw=2)
# Add a big red three to the plot
plt.text(x_coord_number, 4000, 'P2', fontsize=20, color='red')

# Draw an interval for problem 3
interval_x = [x_coord, x_coord]
interval_y = [14500, 26000]
plt.plot(interval_x, interval_y, color='red', lw=2)
plt.plot([interval_x[0] - interval_hor_size, interval_x[0] + interval_hor_size], [interval_y[0], interval_y[0]], color='red', lw=2)
plt.plot([interval_x[1] - interval_hor_size, interval_x[1] + interval_hor_size], [interval_y[1], interval_y[1]], color='red', lw=2)
# Add a big red three to the plot
plt.text(x_coord_number, 19500, 'P3', fontsize=20, color='red')



# Create custom legend handles
marker_handles = [Line2D([0], [0], marker=marker, color='w', markerfacecolor='k', markersize=10, linestyle='None') for marker in complexity_markers.values()]
color_handles = [Line2D([0], [0], color=colormap(i / 10.0), lw=4) for i in range(len(degree_colors))]
minima_handle = [Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=20, linestyle='None')]

# Combine marker and color handles
handles = marker_handles + color_handles + minima_handle
labels = [f'Complexity {k}' for k in complexity_markers.keys()] + [f'Degree {k}' for k in degree_colors.keys()] + ['Minima']

# Create combined legend
plt.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')

plt.savefig('lambda_experiments.eps', format='eps', bbox_inches='tight')
plt.savefig('lambda_experiments.png', format='png', bbox_inches='tight')