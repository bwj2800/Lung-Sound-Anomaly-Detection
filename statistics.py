import matplotlib.pyplot as plt
import numpy as np

# Read the data from the text file
file_path = 'feature/feature_stats_per_class.txt'

data = {
    'normal': {'mean': [], 'std': []},
    'crackle': {'mean': [], 'std': []},
    'wheeze': {'mean': [], 'std': []},
    'both': {'mean': [], 'std': []}
}

with open(file_path, 'r') as f:
    lines = f.readlines()[1:]  # Skip the header line
    for line in lines:
        cls, feature, mean, std = line.strip().split('\t')
        data[cls]['mean'].append(float(mean))
        data[cls]['std'].append(float(std))

# Verify data extraction
for cls in data.keys():
    print(f"{cls} - Mean: {data[cls]['mean'][:5]}, Std: {data[cls]['std'][:5]}")  # Print first 5 values for verification

# Convert lists to numpy arrays
for cls in data.keys():
    data[cls]['mean'] = np.array(data[cls]['mean'])
    data[cls]['std'] = np.array(data[cls]['std'])

# Plot the data
x = np.arange(1, 185)

plt.figure(figsize=(15, 10))

for cls in data.keys():
    plt.plot(x, data[cls]['mean'], label=f'{cls} mean')
    plt.fill_between(x, data[cls]['mean'] - data[cls]['std'], data[cls]['mean'] + data[cls]['std'], alpha=0.2)

plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.title('Feature Means and Standard Deviations per Class')
plt.legend()
plt.savefig('feature_stats_plot.png')
plt.show()