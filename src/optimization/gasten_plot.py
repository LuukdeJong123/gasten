import matplotlib.pyplot as plt
import json
import os
import torch

random_search_data = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores\\param_scores_random_search_step1_8v0_iteration_{i}.pt", map_location=torch.device('cpu'))
    random_search_data.append(data)

# Flatten the data
random_search_scores = [score for iteration in random_search_data for score in iteration.values()]

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

bayesian_optimization_scores = []

directories = get_immediate_subdirectories("C:\\bayesian-8v0\\optimization\\May28T11-06_3tpnccv1")
for directory in directories:
    with open("C:\\bayesian-8v0\\optimization\\May28T11-06_3tpnccv1\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        scores = json_data['eval']['fid']
        bayesian_optimization_scores.extend(scores)



# Plot 1: Convergence Plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(random_search_scores) + 1), sorted(random_search_scores, reverse=True), label='Random Search')
plt.plot(range(1, len(bayesian_optimization_scores) + 1), sorted(bayesian_optimization_scores, reverse=True), label='Bayesian Optimization')
plt.xlabel('Iteration')
plt.ylabel('Best FID Score')
plt.title('Convergence Plot')
plt.legend()
plt.show()

# Plot 2: Box Plot
plt.figure(figsize=(10, 5))
plt.boxplot([random_search_scores, bayesian_optimization_scores], labels=['Random Search', 'Bayesian Optimization'])
plt.ylabel('FID Score')
plt.title('Comparison of FID Scores')
plt.show()

# Plot 3: Scatter Plot
plt.figure(figsize=(10, 5))
plt.scatter(range(1, len(random_search_scores) + 1), random_search_scores, label='Random Search')
plt.scatter(range(1, len(bayesian_optimization_scores) + 1), bayesian_optimization_scores, label='Bayesian Optimization', marker='x')
plt.xlabel('Iteration')
plt.ylabel('FID Score')
plt.title('Iteration vs FID Score')
plt.legend()
plt.show()