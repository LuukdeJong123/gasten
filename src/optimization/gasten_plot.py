import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np

random_search_data = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores_fashion_mnist\\param_scores_random_search_step1_fashion-mnist_3v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    random_search_data.append(data)

# Flatten the data
random_search_scores = [score for iteration in random_search_data for score in iteration.values()]

grid_search_data = []
for i in range(1, 18):
    data = torch.load(f"C:\\grid_search_scores_fashion_mnist\\param_scores_grid_search_step1_fashion-mnist_3v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    grid_search_data.append(data)

# Flatten the data
grid_search_scores = [score for iteration in grid_search_data for score in iteration.values()]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


bayesian_optimization_scores = []

directories = get_immediate_subdirectories("C:\\bayesian_fashion_mnist-3v0\\optimization\\May31T22-54_4ghvj56t")
for directory in directories:
    with open("C:\\bayesian_fashion_mnist-3v0\\optimization\\May31T22-54_4ghvj56t\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        bayesian_optimization_scores.extend(scores)

hyperband_optimization_scores = []

directories = get_immediate_subdirectories("C:\\hyperband_fashion_mnist-3v0\\optimization\\May31T22-55_3p4jan79")
for directory in directories:
    with open("C:\\hyperband_fashion_mnist-3v0\\optimization\\May31T22-55_3p4jan79\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        hyperband_optimization_scores.extend(scores)

BOHB_optimization_scores = []

directories = get_immediate_subdirectories("C:\\BOHB_fashion_mnist-3v0\\optimization\\May31T22-54_24pfsudr")
for directory in directories:
    with open("C:\\BOHB_fashion_mnist-3v0\\optimization\\May31T22-54_24pfsudr\\" + directory + "\\stats.json") as json_file:
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
plt.title('Converging FID Score for fashion-MNIST 3v0 (T-shirt v Dress): Step 1')
plt.legend()
plt.savefig('fasion-MNIST_3v0_convergence_step1.png')

# Plot 2: Box Plot
plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores],
    labels=['Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('FID Score')
plt.title('Boxplot of FID Scores for fashion-MNIST 3v0 (T-shirt v Dress): Step 1')

# Annotate min and max values
def annotate_boxplot(boxplot, data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')

annotate_boxplot(box, [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores])

# Save and show the plot
plt.savefig('fasion-MNIST_3v0_boxplot_step1.png')