import matplotlib.pyplot as plt
import json
import os
from scipy.signal import savgol_filter


def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name)) and name != '__pycache__']


def smooth_data(data, window_size, poly_order=2):
    if window_size <= poly_order:
        poly_order = window_size - 1
    if window_size > len(data):
        window_size = len(data) - 1 if len(data) % 2 == 0 else len(data)
    if window_size <= 1:
        return data  # Return original data if window size is invalid
    return savgol_filter(data, window_size, poly_order)


epochs = []

directories = get_immediate_subdirectories("C:\\May17T15-38_otiyzwl5")
fig, axs = plt.subplots(1, 2, figsize=(24, 12))  # Create subplots side by side and increase the figure size

lines = []
line_styles = ['-', '--', '-.', ':']  # Different line styles
line_width = 2  # Set line width

for i, directory in enumerate(directories):
    with open(
            "C:\\May17T15-38_otiyzwl5\\" + directory + "\\stats.json") as json_file:  # Adjusted for the path
        json_data = json.load(json_file)
        epochs = [*range(1, len(json_data['eval']['fid']) + 1, 1)]
        smoothed_fid = smooth_data(json_data['eval']['fid'], window_size=5)  # Apply stronger smoothing
        smoothed_cd = smooth_data(json_data['eval']['conf_dist'], window_size=5)  # Smooth the data

        line_fid, = axs[0].plot(epochs, smoothed_fid, label="Configuration " + directory,
                                alpha=0.8, linestyle=line_styles[i % len(line_styles)], linewidth=line_width)  # Set alpha for transparency
        line_cd, = axs[1].plot(epochs, smoothed_cd, label="Configuration " + directory,
                               alpha=0.8, linestyle=line_styles[i % len(line_styles)], linewidth=line_width)  # Plot confusion distance
        lines.append(line_fid)

# FID subplot
axs[0].set_xlabel("Epochs", fontsize=16)
axs[0].set_ylabel("FID", fontsize=16)
axs[0].axvline(x=4, linestyle='dashed', color='black')
axs[0].axvline(x=13, linestyle='dashed', color='black')
axs[0].set_xticks(range(1, 42, 2))  # Reduce number of x-ticks
axs[0].tick_params(axis='x', labelsize=14)
axs[0].tick_params(axis='y', labelsize=14)
axs[0].set_title('FID Score Across Epochs for MNIST 8v0 Hyperband optimization: Step 2', fontsize=18)
axs[0].grid(True)  # Add grid

# Confusion Distance subplot
axs[1].set_xlabel("Epochs", fontsize=16)
axs[1].set_ylabel("Confusion Distance", fontsize=16)
axs[1].axvline(x=4, linestyle='dashed', color='black')
axs[1].axvline(x=13, linestyle='dashed', color='black')
axs[1].set_xticks(range(1, 42, 2))  # Reduce number of x-ticks
axs[1].tick_params(axis='x', labelsize=14)
axs[1].tick_params(axis='y', labelsize=14)
axs[1].set_title('Confusion Distance Across Epochs for MNIST 8v0 Hyperband optimization: Step 2', fontsize=18)
axs[1].grid(True)  # Add grid

fig.legend(lines, [line.get_label() for line in lines], loc='lower center',
           fontsize='large', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0))  # Set bbox_to_anchor for better placement

plt.tight_layout(rect=[0, 0.1, 1, 0.95])
plt.savefig('MNIST_8v0_Hyperband_Step_2.png')