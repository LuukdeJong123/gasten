import matplotlib.pyplot as plt

import json
import os


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and name != '__pycache__']


epochs = []

directories = get_immediate_subdirectories("C:\\May17T14-58_1wqj3vdo")
for directory in directories:
    with open(
            "C:\\May17T14-58_1wqj3vdo\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        epochs = [*range(1, len(json_data['eval']['fid']) + 1, 1)]
        plt.plot(epochs, json_data['eval']['fid'], label="Configuration " + directory)

plt.xlabel("Epochs")
plt.ylabel("FID")
plt.axvline(x=2, linestyle='dashed', color='black')
plt.axvline(x=5, linestyle='dashed', color='black')
plt.axvline(x=10, linestyle='dashed', color='black')
plt.xticks(epochs)

plt.legend(loc='center left', bbox_to_anchor=(0.8, 0.8), fontsize='xx-small')
plt.title('FID Score Across Epochs for MNIST 8v0 Hyperband: Step 1')
plt.show()
