import matplotlib.pyplot as plt
import json
import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and name != '__pycache__']

epochs = []

directories = get_immediate_subdirectories("C:\\May17T15-38_otiyzwl5")
plt.figure(figsize=(14, 8))  # Increase figure size
for directory in directories:
    with open(
            "C:\\May17T15-38_otiyzwl5\\" + directory + "\\stats.json") as json_file:
        json_data = json.load(json_file)
        epochs = [*range(1, len(json_data['eval']['fid']) + 1, 1)]
        plt.plot(epochs, json_data['eval']['fid'], label="Configuration " + directory, alpha=0.7)  # Set alpha for transparency

plt.xlabel("Epochs")
plt.ylabel("FID")
plt.axvline(x=4, linestyle='dashed', color='black')
plt.axvline(x=13, linestyle='dashed', color='black')
plt.xticks(range(1, 41, 2))  # Reduce number of x-ticks

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')  # Move legend outside the plot
plt.title('FID Score Across Epochs for MNIST 8v0 Hyperband: Step 1')
plt.tight_layout()  # Adjust layout to fit everything
plt.show()