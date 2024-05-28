import matplotlib.pyplot as plt

import json
import os


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and name != '__pycache__']


epochs = []

directories = get_immediate_subdirectories("C:\\bayesian-8v0\\optimization\\May28T11-06_3tpnccv1")
for directory in directories:
    with open(
            "C:\\bayesian-8v0\\optimization\\May28T11-06_3tpnccv1\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        epochs = [*range(1, len(json_data['eval']['fid']) + 1, 1)]
        plt.plot(epochs, json_data['eval']['fid'], label="Configuration " + directory)

plt.xlabel("Epochs")
plt.ylabel("FID")
# plt.axvline(x=2, linestyle='dashed', color='black')
# plt.axvline(x=5, linestyle='dashed', color='black')
# plt.axvline(x=10, linestyle='dashed', color='black')
plt.xticks(epochs)
plt.grid(True)
plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.8), fontsize='xx-small')
plt.title('FID Score Across Epochs for MNIST 8v0 Bayesian optimization: Step 1')
plt.savefig('MNIST_8v0_Bayesian_Step_1.png')
