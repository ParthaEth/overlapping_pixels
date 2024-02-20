import os
import torch
import torch.optim as optim
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from torchvision import transforms

# Check if CUDA is available and set the default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Load Lena image
lena_path = 'lena.png'  # Update this path accordingly
lena_img = read_image(lena_path).float() / 255  # Normalize to [0, 1]
lena_img = lena_img[:3, ...].to(device)

# Define the resize transformation and apply it
res = 128
resize_transform = transforms.Resize((res, res), antialias=True)
lena_img = resize_transform(lena_img.unsqueeze(0))[0].to(device)  # Add batch dimension for transform

if lena_img.shape[0] == 1:
    lena_img = lena_img.repeat(3, 1, 1)

H, W = lena_img.shape[1], lena_img.shape[2]

# Parameters
N = 400  # Number of Gaussians
colors = torch.randn(N, 3, device=device, requires_grad=True)
means = torch.rand(N, 2, device=device, requires_grad=True)

log_sigma = np.log((1 / (N * 3.14)) ** 0.5)
L_params = (torch.rand(N, 2, 2, device=device) - 0.5) * np.exp(log_sigma)
L_params[:, 0, 0] = log_sigma
L_params[:, 1, 1] = log_sigma
L_params.requires_grad = True

x, y = torch.meshgrid(torch.linspace(0, 0.98, W, device=device), torch.linspace(0, 0.98, H, device=device), indexing='ij')
x, y = x.flatten(), y.flatten()

out_dir = f'results/{lena_path.split(".")[0]}/{res}X{res}/n_gaus:{N}/'
os.makedirs(out_dir, exist_ok=True)

with torch.no_grad():
    for i in range(N):
        mean_pixel = (means[i] * torch.tensor([W-1, H-1], device=device)).long()
        color_sample_radius = max(1, int(np.exp(log_sigma) * max(W, H) / 2))
        sampled_img_area = lena_img[:, max(0, mean_pixel[1] - color_sample_radius):min(H, mean_pixel[1] + color_sample_radius), max(0, mean_pixel[0] - color_sample_radius):min(W, mean_pixel[0] + color_sample_radius)]
        colors[i] = sampled_img_area.reshape(3, -1).mean(dim=1)

def gaussian_2d(x, y, mean, L):
    cov = L @ L.T
    det = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
    inv_cov = torch.tensor([[cov[1, 1], -cov[0, 1]], [-cov[1, 0], cov[0, 0]]], device=device) / det
    pos = torch.stack([x, y], dim=-1)
    diff = pos - mean
    mahalanobis_dist = torch.sum((diff @ inv_cov) * diff, dim=-1)
    return torch.exp(-0.5 * mahalanobis_dist)

optimizer = optim.Adam([means, L_params, colors], lr=0.01)
pbar = tqdm.tqdm(range(10000))

for step in pbar:
    optimizer.zero_grad()
    reconstructed = torch.zeros(3, H, W, device=device)
    for i in range(N):
        L = torch.tril(L_params[i])
        L.diagonal().exp_()
        weight = gaussian_2d(x, y, means[i], L)
        reconstructed += (weight * colors[i, :, None]).view(3, H, W)
    loss = torch.mean((reconstructed - lena_img) ** 2)
    psnr = 10*torch.log10(1/loss)

    loss.backward()
    optimizer.step()
    pbar.set_description(f'comp: {H*W*3/(N*8)}, PSNR {psnr.item():0.2f}, Loss: {loss.item()}')

    if step % 100 == 0:
        # Transfer reconstructed image to CPU for plotting
        current_img_np = reconstructed.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        original_img_np = lena_img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()

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


# recon and save
reconstructed = torch.zeros(1, H * W, device=device)
for i in range(N):
    # Ensure L is a lower triangular matrix with positive diagonals
    L = torch.tril(L_params[i])
    L.diagonal().exp_()
    weight = gaussian_2d(y, x, means[i], L/1.5)  # Use L to compute Gaussian weights
    reconstructed += weight
reconstructed = reconstructed.view(1, H, W)
plt.figure(figsize=(10, 5))
plt.imshow(np.asarray(to_pil_image(reconstructed.detach().cpu().clamp(0, 1))))
plt.title('current image')
plt.axis('off')  # Hide axes ticks

plt.savefig(os.path.join(out_dir, f'reduced_var_gaussians.png'))