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
    subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    subdirs_with_mtime = [(name, os.path.getmtime(os.path.join(directory, name))) for name in subdirs]
    subdirs_sorted_by_age = sorted(subdirs_with_mtime, key=lambda x: x[1])
    return [name for name, mtime in subdirs_sorted_by_age]


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

def load_scores(directory, extension=".pt"):
    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(extension) and 'step1' in filename:
            filepath = os.path.join(directory, filename)
            data = torch.load(filepath)
            for values in data.values():
                if isinstance(values, list):
                    scores.extend(values)
                else:
                    scores.append(values)
    return scores

def load_json_scores(directory):
    scores = []
    sub_subdirectories = get_immediate_subdirectories(directory)
    sub_subdirectory_path = os.path.join(directory, sub_subdirectories[0])
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    scores.extend(json_data['eval']['fid'])
    return scores

directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
random_search_scores = load_scores(directory_rs)

directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
grid_search_scores = load_scores(directory_gs)

bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
bayesian_optimization_scores = load_json_scores(bayesian_directory)

hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
hyperband_optimization_scores = load_json_scores(hyperband_directory)

BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
BOHB_optimization_scores = load_json_scores(BOHB_directory)

plt.figure(figsize=(10, 5))
box = plt.boxplot(
    [flatten_scores(grid_search_scores), flatten_scores(random_search_scores),
     flatten_scores(bayesian_optimization_scores), flatten_scores(hyperband_optimization_scores),
     flatten_scores(BOHB_optimization_scores)], patch_artist=True,
    labels=['Random Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB'])

plt.ylabel('Frechet Inception Distance (FID)')
plt.title(f'Boxplot of FID Scores for {args.dataset} {args.pos_class}v{args.neg_class}: Step 1')


def annotate_boxplot(data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.25, min_val - 10), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.25, max_val + 10), ha='center', color='red')


annotate_boxplot([flatten_scores(grid_search_scores), flatten_scores(random_search_scores),
                  flatten_scores(bayesian_optimization_scores), flatten_scores(hyperband_optimization_scores),
                  flatten_scores(BOHB_optimization_scores)])

plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_boxplot_step1.png')
