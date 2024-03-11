import numpy as np
import torch
from matplotlib import pyplot as plt


def create_exclusive_patches(data, patch_size=5):
    num_points = data.shape[0]
    num_patches = num_points // patch_size
    patches = torch.zeros((num_patches, patch_size, data.shape[1]))
    used = torch.zeros(num_points, dtype=bool)

    # Precompute all pairwise distances
    all_distances = torch.cdist(data, data, p=2)

    # Iterate over each patch
    for i in range(num_patches):
        # Skip already used points
        current_point_index = (used == False).nonzero(as_tuple=True)[0][0]

        # Sort distances for the current point
        sorted_indices = torch.argsort(all_distances[current_point_index])

        # Pick nearest neighbors that are not used
        neighbors = sorted_indices[~used[sorted_indices]][:patch_size]

        # Update the patch
        patches[i] = data[neighbors]

        # Mark these points as used
        used[neighbors] = True

    return patches.reshape(-1, data.shape[1])


if __name__ == "__main__":
    patch_size = 5
    num_patches = 80
    points = torch.rand(num_patches * patch_size, 2)  # Example tensor with random data
    exclusive_patches = create_exclusive_patches(points)
    exclusive_patches = exclusive_patches.reshape(num_patches, patch_size, -1)

    print(exclusive_patches.shape)

    # Plotting
    count = 0
    for patch in exclusive_patches:
        # Generate a random color
        color = np.random.rand(3, )

        # Plot each point in the patch with the same color
        plt.scatter(patch[:, 0], patch[:, 1], c=[color] * patch_size)

        count += 1
        if count > 10:  # Limiting to first 10 patches for clarity
            break

    plt.show()
