import matplotlib.pyplot as plt
import json
import os
import torch
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
        max_key = max(data, key=data.get)
        max_value = data[max_key]
        random_search_scores.append(max_value)

grid_search_scores = []
directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
# Loop through all the files in the directory
for filename in os.listdir(directory_rs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_rs, filename)
        data = torch.load(filepath)
        max_key = max(data, key=data.get)
        max_value = data[max_key]
        grid_search_scores.append(max_value)

bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}"

bayesian_directories = get_immediate_subdirectories(bayesian_directory)
bayesian_directories.sort()
bayesian_second_subdirectory = bayesian_directories[1]

bayesian_second_subdirectory_path = os.path.join(bayesian_directory, bayesian_second_subdirectory)

bayesian_optimization_scores = []
bayesian_stats_file_path = os.path.join(bayesian_second_subdirectory_path, "stats.json")
with open(bayesian_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    best_score = max(scores)
    bayesian_optimization_scores.append(best_score)

hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/"

hyperband_directories = get_immediate_subdirectories(hyperband_directory)
hyperband_directories.sort()
hyperband_second_subdirectory = hyperband_directories[1]

hyperband_second_subdirectory_path = os.path.join(hyperband_directory, hyperband_second_subdirectory)

hpyerband_optimization_scores = []
hpyerband_stats_file_path = os.path.join(hyperband_second_subdirectory_path, "stats.json")
with open(hpyerband_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    best_score = max(scores)
    hpyerband_optimization_scores.extend(best_score)

hyperband_optimization_scores = []

BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/"

BOHB_directories = get_immediate_subdirectories(BOHB_directory)
BOHB_directories.sort()
BOHB_second_subdirectory = BOHB_directories[1]

BOHB_second_subdirectory_path = os.path.join(BOHB_directory, BOHB_second_subdirectory)

BOHB_optimization_scores = []
BOHB_stats_file_path = os.path.join(BOHB_second_subdirectory_path, "stats.json")
with open(BOHB_stats_file_path) as json_file:
    json_data = json.load(json_file)
    scores = json_data['eval']['fid']
    best_score = max(scores)
    BOHB_optimization_scores.extend(best_score)


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
