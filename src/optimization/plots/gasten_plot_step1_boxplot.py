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
    bayesian_optimization_scores.extend(scores)

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
    hpyerband_optimization_scores.extend(scores)

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
    BOHB_optimization_scores.extend(scores)

plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores,
     BOHB_optimization_scores],
    labels=['Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('FID Score')
plt.title('Boxplot of FID Scores for MNIST 8v0: Step 1')


def annotate_boxplot(boxplot, data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')


annotate_boxplot(box,
                 [grid_search_scores, random_search_scores, bayesian_optimization_scores, hyperband_optimization_scores,
                  BOHB_optimization_scores])

plt.savefig('MNIST_8v0_boxplot_step1.png')
