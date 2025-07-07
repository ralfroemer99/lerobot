import numpy as np
import matplotlib.pyplot as plt

from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
)

repo_id = "ralfroemer/pick_lego_block_filtered"

# Copy all frames from the source dataset to the target dataset, but filter out frames with zero actions.
dataset = LeRobotDataset(repo_id=repo_id)

actions_all = np.array(dataset.hf_dataset["action"])    # Shape (N_frames, 7)

# Normalize each action dimension to have zero mean and unit variance
actions_normalized = (actions_all - np.mean(actions_all, axis=0)) / np.std(actions_all, axis=0)

# Create a subplot with 7 subplots, one for each action dimension
_, axs = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
_, axs_normalized = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
for i in range(7):
    # Only dots, no lines
    axs[i].plot(actions_all[:, i], label=f'Action Dimension {i+1}', marker='o', linestyle='None', markersize=2)      
    axs[i].set_ylabel(f'Action {i+1}')
    axs[i].legend()
    axs[i].grid()

    axs_normalized[i].plot(actions_normalized[:, i], label=f'Normalized Action Dimension {i+1}', marker='o', linestyle='None', markersize=2)
    axs_normalized[i].set_ylabel(f'Normalized Action {i+1}')
    axs_normalized[i].legend()
    axs_normalized[i].grid()

plt.show()