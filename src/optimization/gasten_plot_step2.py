import matplotlib.pyplot as plt
import json
import os
import torch

random_search_data = []
for i in range(20):
    data = torch.load(f"C:\\random_search_scores_fashion_mnist\\param_scores_random_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    random_search_data.append(data)

# Flatten the data
random_search_fid_scores = [score[0] for iteration in random_search_data for score in iteration.values()]
random_search_cd_scores = [score[1] for iteration in random_search_data for score in iteration.values()]

grid_search_data = []
for i in range(1, 13):
    data = torch.load(f"C:\\grid_search_scores_fashion_mnist\\param_scores_grid_search_step2_mnist_8v0_iteration_{i}.pt",
                      map_location=torch.device('cpu'))
    grid_search_data.append(data)

# Flatten the data
grid_search_fid_scores = [score[0] for iteration in grid_search_data for score in iteration.values()]
grid_search_cd_scores = [score[1] for iteration in grid_search_data for score in iteration.values()]

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

bayesian_optimization_fid_scores = []
bayesian_optimization_cd_scores = []

directories = get_immediate_subdirectories("C:\\bayesian_mnist-8v0\\optimization\\Jun05T11-21_3un9xyrl")
for directory in directories:
    with open("C:\\bayesian_mnist-8v0\\optimization\\Jun05T11-21_3un9xyrl\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        fid_scores = json_data['eval']['fid']
        cd_scores = json_data['eval']['conf_dist']
        bayesian_optimization_fid_scores.extend(fid_scores)
        bayesian_optimization_cd_scores.extend(cd_scores)

hyperband_optimization_fid_scores = []
hyperband_optimization_cd_scores = []

directories = get_immediate_subdirectories("C:\\hyperband_mnist-8v0\\optimization\\Jun05T11-22_1mm52i0l")
for directory in directories:
    with open("C:\\hyperband_mnist-8v0\\optimization\\Jun05T11-22_1mm52i0l\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        fid_scores = json_data['eval']['fid']
        cd_scores = json_data['eval']['conf_dist']
        hyperband_optimization_fid_scores.extend(fid_scores)
        hyperband_optimization_cd_scores.extend(cd_scores)

BOHB_optimization_fid_scores = []
BOHB_optimization_cd_scores = []

directories = get_immediate_subdirectories("C:\\BOHB_mnist-8v0\\optimization\\Jun05T12-24_5jkoc6m5")
for directory in directories:
    with open("C:\\BOHB_mnist-8v0\\optimization\\Jun05T12-24_5jkoc6m5\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        fid_scores = json_data['eval']['fid']
        cd_scores = json_data['eval']['conf_dist']
        BOHB_optimization_fid_scores.extend(fid_scores)
        BOHB_optimization_cd_scores.extend(cd_scores)

# Plot: Convergence Plots for FID and CD side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# FID Scores
ax1.plot(range(1, len(grid_search_fid_scores) + 1), grid_search_fid_scores, label='Grid search FID', color='b')
ax1.plot(range(1, len(random_search_fid_scores) + 1), random_search_fid_scores, label='Random Search FID', color='g')
ax1.plot(range(1, len(bayesian_optimization_fid_scores) + 1), bayesian_optimization_fid_scores, label='Bayesian Optimization FID', color='r')
ax1.plot(range(1, len(hyperband_optimization_fid_scores) + 1), hyperband_optimization_fid_scores, label='Hyperband FID', color='c')
ax1.plot(range(1, len(BOHB_optimization_fid_scores) + 1), BOHB_optimization_fid_scores, label='BOHB FID', color='m')
ax1.set_xlabel('Iteration')
ax1.set_ylabel('FID Score', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('FID Scores')
ax1.legend(loc='upper right')
ax1.grid(True)

# CD Scores
ax2.plot(range(1, len(grid_search_cd_scores) + 1), grid_search_cd_scores, label='Grid search CD', linestyle='--', color='b')
ax2.plot(range(1, len(random_search_cd_scores) + 1), random_search_cd_scores, label='Random Search CD', linestyle='--', color='g')
ax2.plot(range(1, len(bayesian_optimization_cd_scores) + 1), bayesian_optimization_cd_scores, label='Bayesian Optimization CD', linestyle='--', color='r')
ax2.plot(range(1, len(hyperband_optimization_cd_scores) + 1), hyperband_optimization_cd_scores, label='Hyperband CD', linestyle='--', color='c')
ax2.plot(range(1, len(BOHB_optimization_cd_scores) + 1), BOHB_optimization_cd_scores, label='BOHB CD', linestyle='--', color='m')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Confusion Distance', color='k')
ax2.tick_params(axis='y', labelcolor='k')
ax2.set_title('Confusion Distance Scores')
ax2.legend(loc='upper right')
ax2.grid(True)

fig.suptitle('Converging FID and Confusion Distance Scores for MNIST 8v0: Step 2')

plt.savefig('MNIST_8v0_convergence_step2.png')