import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
import argparse
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', dest='pos_class', default=9,
                        type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=4,
                        type=int, help='Negative class for binary classification')
    parser.add_argument('--dataset', dest='dataset',
                        default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
    return parser.parse_args()


def get_immediate_subdirectories(directory):
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

load_dotenv()
args = parse_args()

random_search_scores = []
directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
# Loop through all the files in the directory
for filename in os.listdir(directory_rs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_rs, filename)
        data = torch.load(filepath)
        random_search_scores.append(data)

grid_search_scores = []
directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
# Loop through all the files in the directory
for filename in os.listdir(directory_rs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_rs, filename)
        data = torch.load(filepath)
        grid_search_scores.append(data)

bayesian_directory = f"{os.environ['FILESDIR']}/tools/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}"

bayesian_directories = get_immediate_subdirectories(bayesian_directory)
bayesian_directories.sort()
bayesian_second_subdirectory = bayesian_directories[1]

bayesian_second_subdirectory_path = os.path.join(bayesian_directory, bayesian_second_subdirectory)

bayesian_optimization_scores = []
bayesian_stats_file_path = os.path.join(bayesian_second_subdirectory_path, "stats.json")
with open(bayesian_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    bayesian_optimization_scores.extend(scores)

hyperband_directory = f"{os.environ['FILESDIR']}/tools/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/"

hyperband_directories = get_immediate_subdirectories(hyperband_directory)
hyperband_directories.sort()
hyperband_second_subdirectory = hyperband_directories[1]

hyperband_second_subdirectory_path = os.path.join(hyperband_directory, hyperband_second_subdirectory)

hpyerband_optimization_scores = []
hpyerband_stats_file_path = os.path.join(hyperband_second_subdirectory_path, "stats.json")
with open(hpyerband_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    hpyerband_optimization_scores.extend(scores)

hyperband_optimization_scores = []

BOHB_directory = f"{os.environ['FILESDIR']}/tools/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/"

BOHB_directories = get_immediate_subdirectories(BOHB_directory)
BOHB_directories.sort()
BOHB_second_subdirectory = BOHB_directories[1]

BOHB_second_subdirectory_path = os.path.join(BOHB_directory, BOHB_second_subdirectory)

BOHB_optimization_scores = []
BOHB_stats_file_path = os.path.join(BOHB_second_subdirectory_path, "stats.json")
with open(BOHB_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    BOHB_optimization_scores.extend(scores)


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
