import numpy as np
import matplotlib.pyplot as plt

# Example data: 300 iterations each with 10 values (sorted in decreasing order)
np.random.seed(0)  # For reproducibility

techniques = {
    "Random Search": np.random.rand(300, 10),
    "Grid Search": np.random.rand(300, 10),
    "Bayesian Optimization": np.random.rand(300, 10),
    "Hyperband": np.random.rand(300, 10),
    "BOHB": np.random.rand(300, 10)
}

# Sort each iteration's values in decreasing order (as per your example data)
for key in techniques:
    techniques[key].sort(axis=1)
    techniques[key] = techniques[key][:, ::-1]

# Create the CDF plot
plt.figure(figsize=(10, 6))

for name, data in techniques.items():
    # Flatten the data and sort
    sorted_data = np.sort(data.flatten())
    # Calculate the CDF
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    # Plot the CDF
    plt.plot(cdf * 100, sorted_data, label=name)

# Add horizontal line at 50%
plt.axhline(y=50, color='r', linestyle='--')

# Labels and title
plt.xlabel('Percentage of Iterations (%)')
plt.ylabel('Performance Threshold (%)')
plt.title('CDF of Comparing HPO Techniques')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
