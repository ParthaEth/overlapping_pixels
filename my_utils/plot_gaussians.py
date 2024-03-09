import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_gaussians(means, L_params, colors, title="Gaussian Distributions"):
    """
    Plots Gaussian distributions as ellipses.

    :param means: Tensor of shape [batch_size, N, 2] representing the means of the Gaussians.
    :param L_params: Tensor of shape [batch_size, N, 2, 2] representing the lower triangular matrix for covariance.
    :param colors: Tensor of shape [batch_size, N, 3] representing RGB colors for each Gaussian.
    :param title: Title of the plot.
    """

    # Calculate covariance matrices
    L = torch.tril(L_params)
    L.diagonal(dim1=-2, dim2=-1).exp_()
    covariances = torch.matmul(L, L.transpose(-2, -1))  # [batch_size, N, 2, 2]

    # Set up the plot
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    for i in range(means.shape[1]):
        mean = means[0, i].detach().numpy()
        covariance = covariances[0, i].detach().numpy()

        # Eigenvalues and eigenvectors for the covariance matrix
        vals, vecs = np.linalg.eigh(covariance)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height of the ellipse from eigenvalues
        width, height = 2 * np.sqrt(vals)

        # Ellipse patch
        ellipse = patches.Ellipse(xy=mean, width=width, height=height, angle=theta, color=colors[0, i].detach().numpy())
        ax.add_patch(ellipse)
        ax.plot(mean[0], mean[1], '*', color=colors[0, i].detach().numpy())
        ellipse.set_clip_box(ax.bbox)
        ellipse.set_alpha(0.5)

    ax.set_aspect('equal')
    plt.show()


if __name__ == "__main__":
    import torch
    # means = torch.tensor([[[0.5, 0.5], [0.5, -0.5]]])
    # L_params = torch.tensor([[[[1, 0], [1, 1]], [[1, 0], [1, 1]]]], dtype=torch.float32)
    # colors = torch.tensor([[[1, 0, 0], [0, 1, 0]]], dtype=torch.float32)
    # data = torch.load('/home/pghosh/repos/overlapping_pixels/results/celeba/128X128/400_gaus/001266.pt',
    data = torch.load('/home/pghosh/repos/overlapping_pixels/results/celeba/128X128/400_gaus/011060.pt',
                      map_location=torch.device('cpu'))
    colors = data['colors'][None, ...]
    colors_normalized = colors - colors.min()
    colors_normalized = colors_normalized / colors_normalized.max()
    means = data['means'][None, ...]
    L_params = data['L_params'][None, ...]
    plot_gaussians(means, L_params, colors_normalized)

    L = torch.tril(L_params)
    L.diagonal(dim1=-2, dim2=-1).exp_()
    cov = torch.matmul(L, L.transpose(-2, -1))

    from convert_images import recon_pix_frm_gaus, normalized_px_y, normalizex_pix_x
    # reconstructed = recon_pix_frm_gaus(normalizex_pix_x, normalized_px_y, means, None, cov, colors)
    reconstructed = recon_pix_frm_gaus(normalizex_pix_x, normalized_px_y, means, L_params, None, colors)
    reco_img = ((reconstructed[0].detach() + 1) / 2).cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imshow(reco_img)
    plt.show()
