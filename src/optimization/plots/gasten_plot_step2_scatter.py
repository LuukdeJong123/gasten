import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
import argparse
from dotenv import load_dotenv
from matplotlib.ticker import ScalarFormatter
import random


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
    scores_fid = []
    scores_cd = []
    for filename in os.listdir(directory):
        if filename.endswith(extension) and 'step2' in filename:
            filepath = os.path.join(directory, filename)
            data = torch.load(filepath)
            for values in data.values():
                scores_fid.append(values[0])
                scores_cd.append(values[1])
    return scores_fid, scores_cd

def load_json_scores(directory):
    scores_fid = []
    scores_cd = []
    sub_subdirectories = get_immediate_subdirectories(directory)
    sub_subdirectory_path = os.path.join(directory, sub_subdirectories[1])
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    scores_fid.extend(json_data['eval']['fid'])
                    scores_cd.extend(json_data['eval']['conf_dist'])
    return scores_fid, scores_cd

directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
random_search_scores, random_search_cd = load_scores(directory_rs)

directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
grid_search_scores, grid_search_cd = load_scores(directory_gs)

bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
bayesian_optimization_scores, bayesian_optimization_cd = load_json_scores(bayesian_directory)

hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
hyperband_optimization_scores, hyperband_optimization_cd = load_json_scores(hyperband_directory)

BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
BOHB_optimization_scores, BOHB_optimization_cd = load_json_scores(BOHB_directory)

methods = ['Random Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB']
colors = ['blue', 'green', 'orange', 'red', 'purple']

all_scores = [
    (grid_search_scores, grid_search_cd),
    (random_search_scores, random_search_cd),
    (bayesian_optimization_scores, bayesian_optimization_cd),
    (hyperband_optimization_scores, hyperband_optimization_cd),
    (BOHB_optimization_scores, BOHB_optimization_cd)
]


def select_random_subset(fid_scores, confusion_distances, subset_size=5000):
    if len(fid_scores) > subset_size:
        indices = random.sample(range(len(fid_scores)), subset_size)
        fid_scores = [fid_scores[i] for i in indices]
        confusion_distances = [confusion_distances[i] for i in indices]
    return fid_scores, confusion_distances

for i, method in enumerate(methods):
    fid_scores, confusion_distances = all_scores[i]
    color = colors[i]

    # Select a random subset of points
    fid_scores, confusion_distances = select_random_subset(fid_scores, confusion_distances)

    # Scatter plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(fid_scores, confusion_distances, color=color, alpha=0.6)
    plt.xlabel('Frechet Inception Distance (FID)')
    plt.ylabel('Average Confusion Distance (CD)')
    plt.title('FID vs. CD for HPO Techniques: Step 2')
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.minorticks_off()
    plt.xticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 60, 70], [0, 5, 10, 15, 20, 25, 30, 35, 40, 60, 70])
    plt.title(f'Scatter Plot: {method}')
    plt.grid(True)
    plt.tight_layout()

    # Hexbin plot
    plt.subplot(1, 2, 2)
    hb = plt.hexbin(fid_scores, confusion_distances, gridsize=20, cmap='Blues')
    plt.xlabel('Frechet Inception Distance (FID)')
    plt.ylabel('Average Confusion Distance (ACD)')
    plt.title(f'Hexbin Plot: {method}')
    plt.grid(True)
    plt.colorbar(hb)
    plt.tight_layout()

    scatter_plot_filename = f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_{method.replace(" ", "_")}_scatter_step2.png'
    plt.savefig(scatter_plot_filename)
