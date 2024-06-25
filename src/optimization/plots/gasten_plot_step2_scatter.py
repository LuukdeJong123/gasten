import matplotlib.pyplot as plt
import json
import os
import torch
import numpy as np
import argparse
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

directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_step2_{args.dataset}.{args.pos_class}v{args.neg_class}"
random_search_scores, random_search_cd = load_scores(directory_rs)

directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_step2_{args.dataset}.{args.pos_class}v{args.neg_class}"
grid_search_scores, grid_search_cd = load_scores(directory_gs)

# bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
# bayesian_optimization_scores, bayesian_optimization_cd = load_json_scores(bayesian_directory)
#
# hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
# hyperband_optimization_scores, hyperband_optimization_cd = load_json_scores(hyperband_directory)
#
# BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
# BOHB_optimization_scores, BOHB_optimization_cd = load_json_scores(BOHB_directory)

methods = ['Random Grid Search', 'Random Search']
colors = ['blue', 'green']

# methods = ['Random Grid Search', 'Random Search', 'Bayesian Optimization', 'Hyperband', 'BOHB']
# colors = ['blue', 'green', 'orange', 'red', 'purple']

all_scores = [
    (grid_search_scores, grid_search_cd),
    (random_search_scores, random_search_cd),
    # (bayesian_optimization_scores, bayesian_optimization_cd),
    # (hyperband_optimization_scores, hyperband_optimization_cd),
    # (BOHB_optimization_scores, BOHB_optimization_cd)
]

plt.figure(figsize=(10, 6))
for i, method in enumerate(methods):
    fid_scores, confusion_distances = all_scores[i]
    plt.scatter(fid_scores, confusion_distances, color=colors[i], label=method, alpha=0.6)

plt.xlabel('Frechet Inception Distance (FID)')
plt.ylabel('Confusion Distance (CD)')
plt.title('FID vs. CD for HPO Techniques: Step 2')
ax = plt.gca()
ax.xaxis.set_major_formatter(ScalarFormatter())
plt.minorticks_off()
plt.xticks([0,5,10,15,20,25,30,35,40,60,70],[0,5,10,15,20,25,30,35,40,60,70])
plt.legend()
plt.grid(True)
plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_scatter_step2.png')
