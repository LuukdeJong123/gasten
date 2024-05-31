import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import random


param_scores = []
for j in range(1, 81):
    for i in range(10):
        param_scores.append(torch.load(f"C:\\grid_search_scores\\param_scores_grid_search_step1_8v0_config_{j}_seed_{i}.pt", map_location=torch.device('cpu')))

# Extract indexes from the first score assuming all have the same structure
indexes = list(param_scores[0].keys())

# Compute means for each configuration (every 10 seeds)
mean_scores = []
for j in range(0, len(param_scores), 10):
    mean_score = {}
    for epoch in indexes:
        mean_score[epoch] = np.mean([param_scores[i][epoch] for i in range(j, j + 10)])
    mean_scores.append(mean_score)

# Function to calculate the average value of a dictionary
def average_value(d):
    return sum(d.values()) / len(d)

# Calculate averages for all dictionaries
averages = [average_value(d) for d in mean_scores]

# Find the index of the dictionary with the highest average value
max_index = averages.index(max(averages))

# Find the index of the dictionary with the lowest average value
min_index = averages.index(min(averages))

# Get the dictionaries with the highest and lowest average values
max_avg_dict = mean_scores[max_index]
min_avg_dict = mean_scores[min_index]

plt.figure(figsize=(12, 8))

# Plot mean scores with higher visibility
for mean_score in mean_scores:
    values = list(mean_score.values())
    if (mean_score == max_avg_dict or mean_score == min_avg_dict):
        plt.plot(indexes, values, linestyle='-', linewidth=3)
    else:
        plt.plot(indexes, values, linestyle='-', linewidth=1, alpha=0.5)


# Select 5 random configurations
random_configs = random.sample(range(1, 81), 5)

# Extract the scores for the selected configurations
random_scores = []
for config in random_configs:
    for seed in range(10):
        random_scores.append((config, torch.load(f"C:\\grid_search_scores\\param_scores_grid_search_step1_8v0_config_{config}_seed_{seed}.pt", map_location=torch.device('cpu'))))

# Create a color map to assign a unique color to each configuration
color_map = plt.cm.get_cmap('tab10', 5)  # Using tab10 colormap for 5 different colors

# Extract indexes from the first score assuming all have the same structure
indexes = list(random_scores[0][1].keys())

# Plot both graphs side by side
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# First subplot: original graph
axes[0].set_title('FID Score for MNIST 8v0 Random Grid Search: Step 1 (80 configurations)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('FID')

for mean_score in mean_scores:
    values = list(mean_score.values())
    if (mean_score == max_avg_dict or mean_score == min_avg_dict):
        axes[0].plot(indexes, values, linestyle='-', linewidth=3)
    else:
        axes[0].plot(indexes, values, linestyle='-', linewidth=1, alpha=0.5)

axes[0].grid(True)
axes[0].yaxis.set_major_locator(MultipleLocator(25))  # Set the tick interval to 25

# Second subplot: new graph with 5 random configurations
axes[1].set_title('FID Score for MNIST 8v0 Random Grid Search: 5 Random Configurations (10 seeds)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('FID')

for i, config in enumerate(random_configs):
    color = color_map(i)  # Get a unique color for the configuration
    for seed in range(10):
        score = random_scores[i * 10 + seed][1]
        values = list(score.values())
        axes[1].plot(indexes, values, linestyle='-', linewidth=1, alpha=0.8, color=color, label=f'Config {config}' if seed == 0 else "")

axes[1].grid(True)
axes[1].yaxis.set_major_locator(MultipleLocator(25))  # Set the tick interval to 25
axes[1].legend(loc='upper right', bbox_to_anchor=(1.3, 1))

# Save and show the combined plot
plt.savefig('MNIST_8v0_grid_search_combined.png', bbox_inches='tight')
plt.show()