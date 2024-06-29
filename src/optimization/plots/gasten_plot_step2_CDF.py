import os
import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
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
                last_value_fid = values[0][len(values[0]) - 1]
                if len(scores_fid) > 0 and last_value_fid > scores_fid[-1]:
                    scores_fid.append(scores_fid[-1])
                else:
                    scores_fid.append(last_value_fid)
                last_value_cd = values[1][len(values[1]) - 1]
                if len(scores_cd) > 0 and last_value_cd > scores_cd[-1]:
                    scores_cd.append(scores_fid[-1])
                else:
                    scores_cd.append(last_value_cd)
    return scores_fid, scores_cd


def load_json_scores(directory):
    scores_fid = []
    scores_cd = []
    sub_subdirectories = get_immediate_subdirectories(directory)
    sub_subdirectory_path = os.path.join(directory, sub_subdirectories[0])
    for root, dirs, files in os.walk(sub_subdirectory_path):
        for file in files:
            if file == "stats.json":
                json_file_path = os.path.join(root, file)
                with open(json_file_path) as json_file:
                    json_data = json.load(json_file)
                    last_score_fid = json_data['eval']['fid'][len(json_data['eval']['fid']) - 1]
                    if len(scores_fid) > 0 and last_score_fid > scores_fid[-1]:
                        scores_fid.append(scores_fid[-1])
                    else:
                        scores_fid.append(last_score_fid)
                    last_score_cd = json_data['eval']['conf_dist'][len(json_data['eval']['conf_dist']) - 1]
                    if len(scores_cd) > 0 and last_score_cd > scores_cd[-1]:
                        scores_cd.append(scores_cd[-1])
                    else:
                        scores_cd.append(last_score_cd)
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

# FID CDF
plt.figure(figsize=(14, 7))
plt.suptitle(f'CDF of HPO Techniques {args.dataset} {args.pos_class}v{args.neg_class}: Step 2', fontsize=16)
plt.subplot(1, 2, 1)
for i, method in enumerate(methods):
    sorted_data = np.sort(np.array(all_scores[i][0]).flatten())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(cdf * 100, sorted_data, label=method, color=colors[i])
plt.ylabel('Frechet Inception Distance (FID)')
plt.xlabel('Percentage of Iterations (%)')
plt.legend()
plt.grid(True)

# Confusion Distance CDF
plt.subplot(1, 2, 2)
for i, method in enumerate(methods):
    sorted_data = np.sort(np.array(all_scores[i][1]).flatten())
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(cdf * 100, sorted_data, label=method, color=colors[i])
plt.ylabel('Confusion Distance (CD)')
plt.xlabel('Percentage of Iterations (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_CDF_step2.png')
