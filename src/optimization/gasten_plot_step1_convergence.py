import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np

random_search_scores = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores_fashion_mnist\\param_scores_random_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    best_score = max(data.values())
    random_search_scores.append(best_score)


grid_search_scores = []
for i in range(1, 13):
    data = torch.load(f"C:\\grid_search_scores_fashion_mnist\\param_scores_grid_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    best_score = max(data.values())
    grid_search_scores.append(best_score)


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


bayesian_optimization_scores = []

directories = get_immediate_subdirectories("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca")
for directory in directories:
    with open("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        best_score = max(scores)
        bayesian_optimization_scores.append(best_score)

hyperband_optimization_scores = []

directories = get_immediate_subdirectories("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in")
for directory in directories:
    with open("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        best_score = max(scores)
        hyperband_optimization_scores.append(best_score)

BOHB_optimization_scores = []

directories = get_immediate_subdirectories("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi")
for directory in directories:
    with open("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        best_score = max(scores)
        BOHB_optimization_scores.append(best_score)


# Plot 1: Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(grid_search_scores) + 1), grid_search_scores, label='Grid search')
plt.plot(range(1, len(random_search_scores) + 1), random_search_scores, label='Random Search')
plt.plot(range(1, len(bayesian_optimization_scores) + 1), bayesian_optimization_scores, label='Bayesian Optimization')
plt.plot(range(1, len(hyperband_optimization_scores) + 1), hyperband_optimization_scores, label='Hyperband')
plt.plot(range(1, len(BOHB_optimization_scores) + 1), BOHB_optimization_scores, label='BOHB')
plt.xlabel('Iteration')
plt.ylabel('FID Score')
plt.title('Converging FID Score for MNIST 8v0: Step 1')
plt.legend()
plt.savefig('MNIST_8v0_convergence_step1.png')
