import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from matplotlib.ticker import ScalarFormatter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', dest='pos_class', default=9,
                        type=int, help='Positive class for binary classification')
    parser.add_argument('--neg', dest='neg_class', default=4,
                        type=int, help='Negative class for binary classification')
    parser.add_argument('--dataset', dest='dataset',
                        default='mnist', help='Dataset (mnist or fashion-mnist or cifar10)')
    return parser.parse_args()


load_dotenv()
args = parse_args()


def annotate_boxplot(data):
    for i, d in enumerate(data, 1):
        min_val = np.min(d)
        max_val = np.max(d)
        plt.annotate(f'{min_val:.2f}', xy=(i, min_val), xytext=(i - 0.1, min_val - 1.5), ha='center', color='blue')
        plt.annotate(f'{max_val:.2f}', xy=(i, max_val), xytext=(i + 0.1, max_val + 1.5), ha='center', color='red')


def get_immediate_subdirectories(directory):
    subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    subdirs_with_mtime = [(name, os.path.getmtime(os.path.join(directory, name))) for name in subdirs]
    subdirs_sorted_by_age = sorted(subdirs_with_mtime, key=lambda x: x[1])
    return [name for name, mtime in subdirs_sorted_by_age]


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

# Boxplot for FID Scores
plt.figure(figsize=(14, 7))
plt.suptitle(f'Boxplot of HPO Techniques {args.dataset} {args.pos_class}v{args.neg_class}: Step 2', fontsize=16)

plt.subplot(1, 2, 1)
fid_data = [scores[0] for scores in all_scores]
plt.boxplot(fid_data, patch_artist=True, labels=methods)
plt.ylabel('Frechet Inception Distance (FID)')
plt.xticks(rotation=15)
annotate_boxplot(fid_data)

plt.subplot(1, 2, 2)
cd_data = [scores[1] for scores in all_scores]
plt.boxplot(cd_data, patch_artist=True, labels=methods)
plt.ylabel('Average Confusion Distance (CD)')
plt.xticks(rotation=15)
annotate_boxplot(cd_data)

plt.tight_layout()

plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_boxplot_step2.png')
