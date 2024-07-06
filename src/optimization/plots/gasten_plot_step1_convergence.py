import matplotlib.pyplot as plt
import json
import os
import torch
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
    subdirs = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    subdirs_with_mtime = [(name, os.path.getmtime(os.path.join(directory, name))) for name in subdirs]
    subdirs_sorted_by_age = sorted(subdirs_with_mtime, key=lambda x: x[1])
    return [name for name, mtime in subdirs_sorted_by_age]


def load_pt_files(directory):
    scores = []
    for filename in os.listdir(directory):
        if filename.endswith(".pt") and 'step1' in filename:
            filepath = os.path.join(directory, filename)
            data = torch.load(filepath)
            last_value = list(data.values())[len(data.keys()) - 1]
            if len(scores) > 0 and last_value > scores[-1]:
                scores.append(scores[-1])
            else:
                scores.append(last_value)
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
                    last_score = json_data['eval']['fid'][len(json_data['eval']['fid']) - 1]
                    if len(scores) > 0 and last_score > scores[-1]:
                        scores.append(scores[-1])
                    else:
                        scores.append(last_score)
    return scores


load_dotenv()
args = parse_args()

directory_rs = f"{os.environ['FILESDIR']}/random_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
random_search_scores = load_pt_files(directory_rs)

directory_gs = f"{os.environ['FILESDIR']}/grid_search_scores_{args.dataset}.{args.pos_class}v{args.neg_class}"
grid_search_scores = load_pt_files(directory_gs)

bayesian_directory = f"{os.environ['FILESDIR']}/out/bayesian_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
bayesian_optimization_scores = load_json_scores(bayesian_directory)

hyperband_directory = f"{os.environ['FILESDIR']}/out/hyperband_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
hyperband_optimization_scores = load_json_scores(hyperband_directory)

BOHB_directory = f"{os.environ['FILESDIR']}/out/BOHB_{args.dataset}-{args.pos_class}v{args.neg_class}/optimization"
BOHB_optimization_scores = load_json_scores(BOHB_directory)

print(min(random_search_scores))
print(min(grid_search_scores))
print(min(bayesian_optimization_scores))
print(min(hyperband_optimization_scores))
print(min(BOHB_optimization_scores))

# Plot 1: Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(grid_search_scores) + 1), grid_search_scores, label='Random Grid search', color='blue')
plt.plot(range(1, len(random_search_scores) + 1), random_search_scores, label='Random Search', color='green')
plt.plot(range(1, len(bayesian_optimization_scores) + 1), bayesian_optimization_scores, label='Bayesian Optimization', color='orange')
plt.plot(range(1, len(hyperband_optimization_scores) + 1), hyperband_optimization_scores, label='Hyperband', color='Red')
plt.plot(range(1, len(BOHB_optimization_scores) + 1), BOHB_optimization_scores, label='BOHB', color='purple')
plt.xlabel('Iteration')
plt.ylabel('FID Score')
plt.yscale('log')
ax = plt.gca()
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_minor_formatter(ScalarFormatter())
plt.minorticks_off()
plt.yticks([5,10,20,50,100,300,500,700],[5,10,20,50,100,300,500,700])
plt.title(f'Converging FID Score for {args.dataset} {args.pos_class}v{args.neg_class}: Step 1')
plt.legend()
plt.grid(True)
plt.savefig(f'{os.environ["FILESDIR"]}/images/{args.dataset}_{args.pos_class}v{args.neg_class}_convergence_step1.png')
