import os
import torch
import torch.optim as optim
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from convert_celebA import celeba_loader


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define batch size, image resolution (H, W), and number of Gaussians (N)
batch_size = 40
res = 128  # Example resolution
H, W = res, res
N = 400  # Number of Gaussians per image
print(f'Compression: {H * W * 3 /(N *8):.2f}')

# Initialize parameters for a batch of images
# Colors: [batch_size, N, 3], Means: [batch_size, N, 2], L_params: [batch_size, N, 2, 2]
colors = torch.zeros(batch_size, N, 3, device=device, requires_grad=True)
means = torch.rand(batch_size, N, 2, device=device, requires_grad=True)
# Convert means to pixel positions
mean_pixels = (means * torch.tensor([W-1, H-1], device=device)).long()

log_sigma = np.log((1 / (N * 3.14)) ** 0.5)
L_params = (torch.rand(batch_size, N, 2, 2, device=device) - 0.5) * np.exp(log_sigma)
L_params[:, :, 0, 0] = log_sigma
L_params[:, :, 1, 1] = log_sigma
L_params.requires_grad = True

train_loader, valid_loader, test_loader = celeba_loader.get_celeba_data_loaders(batch_size)

for lena_images, _ in train_loader:
    imgs = lena_images.to(device)
    x, y = torch.meshgrid(torch.linspace(0, 1, W, device=device), torch.linspace(0, 1, H, device=device), indexing='ij')
    x, y = x.flatten(), y.flatten()

    out_dir = f'results/celeba/{res}X{res}/{N}_gaus/'
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        radius = max(1, int(np.exp(log_sigma) * max(W, H) / 2))
        for i in range(N):
            mean_pixel = mean_pixels[:, i, :]
            x_start = torch.clamp(mean_pixel[:, 0] - radius, 0, W - 1)
            x_end = torch.clamp(mean_pixel[:, 0] + radius + 1, 0, W)
            y_start = torch.clamp(mean_pixel[:, 1] - radius, 0, H - 1)
            y_end = torch.clamp(mean_pixel[:, 1] + radius + 1, 0, H)

            # Sample and compute mean color for each Gaussian, across all images in the batch
            for j in range(batch_size):
                colors[j, i, :] = imgs[j, :, y_start[j]:y_end[j], x_start[j]:x_end[j]].mean(dim=(-2, -1))



    def gaussian_2d_batch(x, y, means, L_params):
        batch_size, N, _ = means.shape
        HW = x.shape[0]  # Total number of points in the grid (H*W)

        # Expand x and y for broadcasting across batches and Gaussians
        x_expanded = x.view(1, -1, 1).expand(batch_size, -1, N)  # Shape: [batch_size, HW, N]
        y_expanded = y.view(1, -1, 1).expand(batch_size, -1, N)  # Shape: [batch_size, HW, N]

        # Expand means for broadcasting across the HW dimension
        means_expanded = means.unsqueeze(1).expand(-1, HW, -1, -1)  # Shape: [batch_size, HW, N, 2]

        # Compute the differences for x and y coordinates separately
        diff_x = x_expanded - means_expanded[..., 0]  # Shape: [batch_size, HW, N]
        diff_y = y_expanded - means_expanded[..., 1]  # Shape: [batch_size, HW, N]

        # Process L_params to compute the inverse covariance matrix for each Gaussian
        L = torch.tril(L_params)
        L.diagonal(dim1=-2, dim2=-1).exp_()
        inv_L = torch.inverse(L)  # Shape: [batch_size, N, 2, 2]
        inv_cov = torch.matmul(inv_L, inv_L.transpose(-2, -1))  # Shape: [batch_size, N, 2, 2]

        # Compute the squared Mahalanobis distance for each point and each Gaussian
        mahalanobis_dist = (inv_cov[:, :, 0, 0].unsqueeze(1) * diff_x ** 2 +
                            (inv_cov[:, :, 0, 1] + inv_cov[:, :, 1, 0]).unsqueeze(1) * diff_x * diff_y +
                            inv_cov[:, :, 1, 1].unsqueeze(1) * diff_y ** 2)  # Shape: [batch_size, HW, N]

        # Compute weights using the Mahalanobis distance
        weights = torch.exp(-0.5 * mahalanobis_dist)

        return weights.transpose(1, 2)  # Transpose to match expected shape [batch_size, N, HW]



    optimizer = optim.Adam([means, L_params, colors], lr=0.01)
    pbar = tqdm.tqdm(range(10000))

    for step in pbar:
        optimizer.zero_grad()

        # Parallel computation of weights for all Gaussians
        weights = gaussian_2d_batch(y, x, means, L_params)  # Shape: (batch, N, H*W)

        # Compute the weighted sum of colors for each point
        colors_expanded = colors.unsqueeze(3).expand(-1, -1, -1, H * W)  # Shape: (b, N, 3, H*W)
        reconstructed = torch.sum(weights.unsqueeze(2) * colors_expanded, dim=1).view(batch_size, 3, H, W)

        loss = torch.mean((reconstructed - imgs) ** 2)
        psnr = 10 * torch.log10((imgs.max() - imgs.min()) / loss)

        loss.backward()
        optimizer.step()
        pbar.set_description(f'PSNR {psnr.item():0.2f}, Loss: {loss.item()}')

        if step % 100 == 0:
            # Transfer reconstructed image to CPU for plotting
            current_img_np = ((reconstructed[0].detach() + 1 )/2).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
            original_img_np = ((imgs[0].detach() + 1)/2).cpu().clamp(0, 1).permute(1, 2, 0).numpy()

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img_np)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(current_img_np)
            plt.title('Reconstructed Image')
            plt.axis('off')

            plt.savefig(os.path.join(out_dir, f'step_{step}_psnr_{psnr.item():.4f}.png'))
            plt.close()  # Close the figure to free memory


    # # recon and save
    # reconstructed = torch.zeros(1, H * W, device=device)
    # for i in range(N):
    #     # Ensure L is a lower triangular matrix with positive diagonals
    #     L = torch.tril(L_params[i])
    #     L.diagonal().exp_()
    #     weight = gaussian_2d(y, x, means[i], L/1.5)  # Use L to compute Gaussian weights
    #     reconstructed += weight
    # reconstructed = reconstructed.view(1, H, W)
    # plt.figure(figsize=(10, 5))
    # plt.imshow(np.asarray(to_pil_image(reconstructed.detach().cpu().clamp(0, 1))))
    # plt.title('current image')
    # plt.axis('off')  # Hide axes ticks
    #
    # plt.savefig(os.path.join(out_dir, f'reduced_var_gaussians.png'))