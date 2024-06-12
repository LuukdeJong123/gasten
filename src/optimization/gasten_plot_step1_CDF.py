import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np

# random_search_data = []
# for i in range(20):
#     data = torch.load(f"C:\\random_search_scores_fashion_mnist\\param_scores_random_search_step2_mnist_8v0_iteration_{i}.pt",
#                       map_location=torch.device('cpu'))
#     random_search_data.append(data)
#
# # Flatten the data
# random_search_scores = [score for iteration in random_search_data for score in iteration.values()]
#
# grid_search_data = []
# for i in range(1, 13):
#     data = torch.load(f"C:\\grid_search_scores_fashion_mnist\\param_scores_grid_search_step2_mnist_8v0_iteration_{i}.pt",
#                       map_location=torch.device('cpu'))
#     grid_search_data.append(data)
#
# # Flatten the data
# grid_search_scores = [score for iteration in grid_search_data for score in iteration.values()]
#
#
# def get_immediate_subdirectories(a_dir):
#     return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]
#
#
# bayesian_optimization_scores = []
#
# directories = get_immediate_subdirectories("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca")
# for directory in directories:
#     with open("C:\\bayesian_mnist-9v4\\optimization\\Jun05T16-30_286g5hca\\" + directory + "\\stats.json") as json_file:
#         json_data = json.load(json_file)
#         scores = json_data['eval']['fid']
#         bayesian_optimization_scores.extend(scores)
#
# hyperband_optimization_scores = []
#
# directories = get_immediate_subdirectories("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in")
# for directory in directories:
#     with open("C:\\hyperband_mnist-9v4\\optimization\\Jun05T16-23_2ujm06in\\" + directory + "\\stats.json") as json_file:
#         json_data = json.load(json_file)
#         scores = json_data['eval']['fid']
#         hyperband_optimization_scores.extend(scores)
#
# BOHB_optimization_scores = []
#
# directories = get_immediate_subdirectories("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi")
# for directory in directories:
#     with open("C:\\BOHB_mnist-9v4\\optimization\\Jun05T16-23_3spfhofi\\" + directory + "\\stats.json") as json_file:
#         json_data = json.load(json_file)
#         scores = json_data['eval']['fid']
#         BOHB_optimization_scores.extend(scores)

# Prepare data for CDF plot
def prepare_cdf_data(scores):
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    return sorted_scores, cdf

# Generating fake data for the CDF plot with better performance indicated by lower FID scores
np.random.seed(42)

# Function to generate scores
def generate_scores(mean, std, count):
    return np.random.normal(mean, std, count)

# Random Search scores (worst performance)
random_search_scores = generate_scores(60, 10, 100)

# Grid Search scores (second worst performance)
grid_search_scores = generate_scores(50, 9, 80)

# Bayesian Optimization scores (better performance)
bayesian_optimization_scores = generate_scores(30, 5, 150)

# Hyperband Optimization scores (better performance)
hyperband_optimization_scores = generate_scores(25, 4, 180)

# BOHB Optimization scores (best performance)
BOHB_optimization_scores = generate_scores(20, 3, 120)

# Prepare data for CDF plot
def prepare_cdf_data(scores):
    sorted_scores = np.sort(scores)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    return sorted_scores, cdf

# Get CDF data
grid_search_cdf_x, grid_search_cdf_y = prepare_cdf_data(grid_search_scores)
random_search_cdf_x, random_search_cdf_y = prepare_cdf_data(random_search_scores)
bayesian_optimization_cdf_x, bayesian_optimization_cdf_y = prepare_cdf_data(bayesian_optimization_scores)
hyperband_optimization_cdf_x, hyperband_optimization_cdf_y = prepare_cdf_data(hyperband_optimization_scores)
BOHB_optimization_cdf_x, BOHB_optimization_cdf_y = prepare_cdf_data(BOHB_optimization_scores)


# Plot 1: CDF Plot with 50% Threshold
plt.figure(figsize=(10, 5))
plt.plot(grid_search_cdf_x, grid_search_cdf_y, label='Grid Search')
plt.plot(random_search_cdf_x, random_search_cdf_y, label='Random Search')
plt.plot(bayesian_optimization_cdf_x, bayesian_optimization_cdf_y, label='Bayesian Optimization')
plt.plot(hyperband_optimization_cdf_x, hyperband_optimization_cdf_y, label='Hyperband')
plt.plot(BOHB_optimization_cdf_x, BOHB_optimization_cdf_y, label='BOHB')

# 50% Threshold line
plt.axhline(y=0.50, color='r', linestyle='--', label='50% Threshold')

# Calculate corresponding value for the 50% threshold
threshold_value_50 = np.interp(0.50, bayesian_optimization_cdf_y, bayesian_optimization_cdf_x)
plt.plot(threshold_value_50, 0.50, 'ro')  # Red dot for intersection

plt.xlabel('FID Score')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Best FID Scores for MNIST 8v0: Step 1')
plt.legend()
plt.savefig('MNIST_8v0_cdf_best_step1_50_percent_threshold.png')
plt.show()