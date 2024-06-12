import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np

random_search_data = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores_fashion_mnist\\param_scores_random_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    random_search_data.append(data)

# Flatten the data
random_search_scores = [score for iteration in random_search_data for score in iteration.values()]

grid_search_data = []
for i in range(1, 13):
    data = torch.load(f"C:\\grid_search_scores_fashion_mnist\\param_scores_grid_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    grid_search_data.append(data)

# Flatten the data
grid_search_scores = [score for iteration in grid_search_data for score in iteration.values()]


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


bayesian_optimization_scores = []

directories = get_immediate_subdirectories("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca")
for directory in directories:
    with open("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        bayesian_optimization_scores.extend(scores)

hyperband_optimization_scores = []

directories = get_immediate_subdirectories("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in")
for directory in directories:
    with open("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        hyperband_optimization_scores.extend(scores)

BOHB_optimization_scores = []

directories = get_immediate_subdirectories("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi")
for directory in directories:
    with open("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        BOHB_optimization_scores.extend(scores)


plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores],
    labels=['Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('FID Score')
plt.title('Boxplot of FID Scores for MNIST 8v0: Step 1')

def annotate_boxplot(boxplot, data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')

annotate_boxplot(box, [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores, BOHB_optimization_scores])

plt.savefig('MNIST_8v0_boxplot_step1.png')