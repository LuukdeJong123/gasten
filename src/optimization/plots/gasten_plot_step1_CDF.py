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
        random_search_scores.append(list(data.values()))

grid_search_scores = []
directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
# Loop through all the files in the directory
for filename in os.listdir(directory_gs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_gs, filename)
        data = torch.load(filepath)
        grid_search_scores.append(list(data.values()))

print(grid_search_scores[0])

bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
bayesian_directories = get_immediate_subdirectories(bayesian_directory)
bayesian_second_subdirectory_path = os.path.join(bayesian_directory, bayesian_directories[0])
bayesian_sub_subdirectories = get_immediate_subdirectories(bayesian_second_subdirectory_path)

bayesian_optimization_scores = []
for sub_subdirectory in bayesian_sub_subdirectories:
    sub_subdirectory_path = os.path.join(bayesian_second_subdirectory_path, sub_subdirectory)
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    scores = json_data['eval']['fid']
                    bayesian_optimization_scores.append(scores)

hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
hyperband_directories = get_immediate_subdirectories(hyperband_directory)
hyperband_second_subdirectory_path = os.path.join(hyperband_directory, hyperband_directories[0])
hyperband_sub_subdirectories = get_immediate_subdirectories(hyperband_second_subdirectory_path)

hyperband_optimization_scores = []
for sub_subdirectory in hyperband_sub_subdirectories:
    sub_subdirectory_path = os.path.join(hyperband_second_subdirectory_path, sub_subdirectory)
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    scores = json_data['eval']['fid']
                    hyperband_optimization_scores.append(scores)

BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
BOHB_directories = get_immediate_subdirectories(BOHB_directory)
BOHB_second_subdirectory_path = os.path.join(BOHB_directory, BOHB_directories[0])
BOHB_sub_subdirectories = get_immediate_subdirectories(BOHB_second_subdirectory_path)

BOHB_optimization_scores = []
for sub_subdirectory in BOHB_sub_subdirectories:
    sub_subdirectory_path = os.path.join(BOHB_second_subdirectory_path, sub_subdirectory)
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    scores = json_data['eval']['fid']
                    BOHB_optimization_scores.append(scores)


def compute_cdf(data):
    data_sorted = np.sort(data)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
    threshold_value = np.percentile(data_sorted, 50)  # 50% threshold
    percentage_above_threshold = 1 - cdf[np.searchsorted(data_sorted, threshold_value)]
    return data_sorted, percentage_above_threshold

plt.figure(figsize=(10, 6))

def plot_results(results, label):
    for result in results:
        sorted_data, percentage_above_threshold = compute_cdf(result)
        plt.plot(sorted_data, np.full_like(sorted_data, percentage_above_threshold), label=f'{label}')


plot_results(random_search_scores, 'Random Search')
plot_results(grid_search_scores, 'Grid Search')
plot_results(bayesian_optimization_scores, 'Bayesian Optimization')
plot_results(hyperband_optimization_scores, 'Hyperband')
plot_results(BOHB_optimization_scores, 'BOHB')

plt.title('CDF of HPO Techniques')
plt.xlabel('Objective Function Value')
plt.ylabel('Cumulative Probability')
plt.legend()
plt.grid(True)
plt.savefig('MNIST_8v0_CDF_step1.png')
