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
            last_value = list(data.values())[len(data.keys()) - 1]
            scores_fid.append(last_value[0])
            scores_cd.append(last_value[1])
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
                    last_score_fid = json_data['eval']['fid'][len(json_data['eval']['fid']) - 1]
                    scores_fid.append(last_score_fid)
                    last_score_cd = json_data['eval']['conf_dist'][len(json_data['eval']['conf_dist']) - 1]
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


def find_pareto_front(scores_fid, scores_cd):
    points = np.array(list(zip(scores_fid, scores_cd)))
    pareto_front = np.ones(points.shape[0], dtype=bool)

    for i, point in enumerate(points):
        for j, other_point in enumerate(points):
            if i != j:
                if (other_point[0] <= point[0] and other_point[1] < point[1]) or \
                        (other_point[0] < point[0] and other_point[1] <= point[1]):
                    pareto_front[i] = 0
                    break

    return points[pareto_front]

pareto_front = find_pareto_front(bayesian_optimization_scores, bayesian_optimization_cd)
plt.scatter(bayesian_optimization_scores, bayesian_optimization_cd, label='All Points')
plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='r', label='Pareto Front')
plt.xlabel('FID')
plt.ylabel('CD')
plt.legend()
plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_pareto_front_step2.png')
print(pareto_front)
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
for i, method in enumerate(methods):
    sorted_fid = np.sort(np.array(all_scores[i][0]).flatten())
    cdf = np.arange(1, len(sorted_fid) + 1) / len(sorted_fid)
    plt.plot(sorted_fid, cdf * 100, label=method, color=colors[i])
plt.xlabel('Frechet Inception Distance (FID)')
plt.ylabel('Percentage of Iterations (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_CDF_step2.png')
