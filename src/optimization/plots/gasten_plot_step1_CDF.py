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
    return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

load_dotenv()
args = parse_args()

def load_scores(directory, extension=".pt"):
    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
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
    for sub_subdirectory in get_immediate_subdirectories(directory):
        sub_subdirectory_path = os.path.join(directory, sub_subdirectory)
        for root, dirs, files in os.walk(sub_subdirectory_path):
            for file in files:
                if file == "stats.json":
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path) as json_file:
                        json_data = json.load(json_file)
                        scores.extend(json_data['eval']['fid'])
    return scores

# Load Random Search Scores
directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
random_search_scores = load_scores(directory_rs)

# Load Grid Search Scores
directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
grid_search_scores = load_scores(directory_gs)

# Load Bayesian Optimization Scores
bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
bayesian_optimization_scores = load_json_scores(bayesian_directory)

# Load Hyperband Scores
hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
hyperband_optimization_scores = load_json_scores(hyperband_directory)

# Load BOHB Scores
BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
BOHB_optimization_scores = load_json_scores(BOHB_directory)

# Prepare data for CDF plotting
techniques = {
    "Random Search": np.array(random_search_scores),
    "Grid Search": np.array(grid_search_scores),
    "Bayesian Optimization": np.array(bayesian_optimization_scores),
    "Hyperband": np.array(hyperband_optimization_scores),
    "BOHB": np.array(BOHB_optimization_scores)
}

# Create the CDF plot with reversed axes
plt.figure(figsize=(10, 6))

for name, data in techniques.items():
    if len(data) == 0:
        continue
    # Flatten the data and sort
    sorted_data = np.sort(np.array(data).flatten())
    # Calculate the CDF
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Plot the CDF with reversed axes
    plt.plot(sorted_data, cdf * 100, label=name)

# Labels and title
plt.ylabel('Percentage of Iterations (%)')
plt.xlabel('FID')
plt.title('CDF of Comparing HPO Techniques')
plt.legend()
plt.grid(True)

plt.savefig(f'{args.dataset}_{args.pos_class}v{args.neg_class}_CDF_step1.png')


plt.figure(figsize=(12, 8))

for name, data in techniques.items():
    if len(data) == 0:
        continue
    plt.hist(np.array(data).flatten(), bins=30, alpha=0.5, label=name)

plt.xlabel('FID')
plt.ylabel('Frequency')
plt.title('Distribution of Performance Scores')
plt.legend()
plt.savefig(f'{args.dataset}_{args.pos_class}v{args.neg_class}_histogram_step1.png')