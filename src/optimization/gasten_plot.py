import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np

random_search_data = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores\\param_scores_random_search_step1_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    random_search_data.append(data)

# Flatten the data
random_search_scores = [score for iteration in random_search_data for score in iteration.values()]

grid_search_data = []
for i in range(1, 20):
    data = torch.load(f"C:\\grid_search_scores_tmp\\param_scores_grid_search_step1_8v0_iteration{i}.pt",
                      map_location=torch.device('cpu'))
    grid_search_data.append(data)

# Flatten the data
grid_search_scores = [score for iteration in grid_search_data for score in iteration.values()]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


bayesian_optimization_scores = []

directories = get_immediate_subdirectories("C:\\May31T08-59_2k23k1z3")
for directory in directories:
    with open("C:\\May31T08-59_2k23k1z3\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        bayesian_optimization_scores.extend(scores)

hyperband_optimization_scores = []

directories = get_immediate_subdirectories("C:\\May31T08-59_3rx9915r")
for directory in directories:
    with open("C:\\May31T08-59_3rx9915r\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        hyperband_optimization_scores.extend(scores)

BOHB_optimization_scores = []

directories = get_immediate_subdirectories("C:\\May31T09-00_2c5p03kx")
for directory in directories:
    with open("C:\\May31T09-00_2c5p03kx\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        BOHB_optimization_scores.extend(scores)

# Plot 1: Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(grid_search_scores) + 1), sorted(grid_search_scores, reverse=True), label='Grid search')
plt.plot(range(1, len(random_search_scores) + 1), sorted(random_search_scores, reverse=True), label='Random Search')
plt.plot(range(1, len(bayesian_optimization_scores) + 1), sorted(bayesian_optimization_scores, reverse=True),
         label='Bayesian Optimization')
plt.plot(range(1, len(hyperband_optimization_scores) + 1), sorted(hyperband_optimization_scores, reverse=True),
         label='Hyperband')
plt.plot(range(1, len(BOHB_optimization_scores) + 1), sorted(BOHB_optimization_scores, reverse=True), label='BOHB')
plt.xlabel('Iteration')
plt.ylabel('FID Score')
plt.title('Converging FID Score for MNIST 8v0: Step 1')
plt.legend()
plt.savefig('MNIST_8v0_convergence_step1.png')

# Plot 2: Box Plot
plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores],
    labels=['Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('FID Score')
plt.title('Boxplot of FID Scores for MNIST 8v0: Step 1')

# Annotate min and max values
def annotate_boxplot(boxplot, data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')

annotate_boxplot(box, [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores])

# Save and show the plot
plt.savefig('MNIST_8v0_boxplot_step1.png')