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


def flatten_scores(scores):
    flattened = []
    for sublist in scores:
        if isinstance(sublist, (list, np.ndarray)):
            flattened.extend(sublist)
        else:
            flattened.append(sublist)
    return flattened


load_dotenv()
args = parse_args()

random_search_scores = []
directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
for filename in os.listdir(directory_rs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_rs, filename)
        data = torch.load(filepath)
        random_search_scores.extend(data.values())

grid_search_scores = []
directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
for filename in os.listdir(directory_gs):
    if filename.endswith(".pt"):
        filepath = os.path.join(directory_gs, filename)
        data = torch.load(filepath)
        grid_search_scores.extend(data.values())

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

plt.figure(figsize=(10, 5))

plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [flatten_scores(grid_search_scores), flatten_scores(random_search_scores),
     flatten_scores(bayesian_optimization_scores), flatten_scores(hyperband_optimization_scores),
     flatten_scores(BOHB_optimization_scores)],
    labels=['Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('Frechet Inception Distance (FID)')
plt.title(f'Boxplot of FID Scores for {args.dataset} {args.pos_class}v{args.neg_class}: Step 1')


def annotate_boxplot(boxplot, data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')


annotate_boxplot(box,
                 [flatten_scores(grid_search_scores), flatten_scores(random_search_scores),
                  flatten_scores(bayesian_optimization_scores), flatten_scores(hyperband_optimization_scores),
                  flatten_scores(BOHB_optimization_scores)])

plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_boxplot_step1.png')
