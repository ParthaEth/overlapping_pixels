import shutil
import sys

import numpy as np

sys.path.append('..')
import os
import csv
import torch
import configs
from collections import namedtuple
import tqdm
from torch.utils.data import Dataset, DataLoader
from convert_celebA.extract_patches import create_exclusive_patches

CSV = namedtuple("CSV", ["header", "index", "data"])


class GaussianPixelDataset(Dataset):
    def __init__(self, root_dir, split, normalize_gaus_params=False, cache_path='/dev/shm'):
        """
        Args:
            gaussian_pix_dir (string): Directory with all the .pt files.
        """
        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        self.cache_path = cache_path
        self.normalize_gaus_params = normalize_gaus_params
        # mean_x, mean_y, covar_xx, covar_xy, covar_yy, r, g, b
        self.max_feat = torch.tensor([2.5512, 2.7300, 1609.8877, 23.1062, 715.4991, 8.3924, 8.2861, 8.2866])
        self.min_feat = torch.tensor([-1.6376e+00, -1.1794e+00, 2.2462e-09, -1.6270e+01, 5.0072e-07, -9.8935e+00, -9.1950e+00, -9.2017e+00])
        self.feat_trunc = 5
        self.max_feat = torch.clip(self.max_feat, -self.feat_trunc, self.feat_trunc)
        self.min_feat = torch.clip(self.min_feat, -self.feat_trunc, self.feat_trunc)
        self.root = root_dir
        self.base_folder = 'celeba'
        # self.gaussian_pix_dir = os.path.join(self.root, 'gps')
        self.gaussian_pix_dir = os.path.join(self.root, 'gps_constrained')
        self.all_file_names = [f for f in sorted(os.listdir(self.gaussian_pix_dir)) if f.endswith('.pt')]
        attr = self._load_csv("list_attr_celeba.txt", header=1)
        # map from {-1, 1} to {0, 1}
        attr = torch.div(attr.data + 1, 2, rounding_mode="floor")
        split_ = split_map[split.lower()]
        splits = self._load_csv("list_eval_partition.txt")

        mask = slice(None) if split_ is None else (splits.data == split_).squeeze()

        all_file_names_set = set(self.all_file_names)

        # Preallocate memory for arrays to avoid resizing
        num_files = len(splits.index)
        self.filenames_this_split = [0 for _ in range(num_files)]
        self.attr = [0 for _ in range(num_files)]

        # Get the intersection of index_set and self.all_file_names
        # valid_files = index_set.intersection(self.all_file_names)

        count_valid_files = 0
        for i, curr_file in enumerate(tqdm.tqdm(splits.index)):
            if mask != slice(None) and not mask[i]:
                continue
            gp_rep = os.path.splitext(curr_file)[0] + '.pt'
            if gp_rep in all_file_names_set:
                self.filenames_this_split[count_valid_files] = gp_rep
                self.attr[count_valid_files] = attr.data[i]
                count_valid_files += 1

        self.filenames_this_split[count_valid_files:] = []
        self.attr[count_valid_files:] = []

    def __len__(self):
        return len(self.filenames_this_split)

    def _load_csv(self,
                  filename: str,
                  header=None,):
        with open(os.path.join(self.root, self.base_folder, filename)) as csv_file:
            data = list(csv.reader(csv_file, delimiter=" ", skipinitialspace=True))

        if header is not None:
            headers = data[header]
            data = data[header + 1:]
        else:
            headers = []

        indices = [row[0] for row in data]
        data = [row[1:] for row in data]
        data_int = [list(map(int, i)) for i in data]

        return CSV(headers, indices, torch.tensor(data_int))

    @torch.no_grad()
    def __getitem__(self, idx):
        if self.cache_path:
            cache_file_path = os.path.join(self.cache_path, self.filenames_this_split[idx])
            if os.path.exists(cache_file_path):
                data = torch.load(cache_file_path, map_location=torch.device('cpu'))
            else:
                file_path = os.path.join(self.gaussian_pix_dir, self.filenames_this_split[idx])
                data = torch.load(file_path, map_location=torch.device('cpu'))
                shutil.copy(file_path, cache_file_path)

        means_in_pm_1 = data['means'] * 2 - 1 + torch.clip(torch.randn(data['means'].shape) * (1/64), -2/64, 2/64)

        # Concatenate cvar from L_params
        # L = torch.tril(data['L_params'] * 0 - torch.eye(2, device=torch.device('cpu')) * 5)
        L = torch.tril(data['L_params'])
        L.diagonal(dim1=-2, dim2=-1).exp_()

        covar = (L@L.transpose(1, 2))# zeroing this feature out to see if that effects accuracy

        # Compute the eigenvalues and eigenvectors for each covariance matrix
        vals, vecs = torch.linalg.eigh(covar + torch.eye(2, device=covar.device)*1e-6)

        # Sort the eigenvalues and eigenvectors in descending order for each matrix in the batch
        order = torch.argsort(vals, dim=-1, descending=True)

        # Use advanced indexing to sort the eigenvalues and eigenvectors
        order_expanded = order.unsqueeze(-1).expand(-1, -1, 2)
        vals = torch.gather(vals, -1, order)
        vecs = torch.gather(vecs, -2, order_expanded)

        # Calculate the angles in degrees for each matrix in the batch
        theta = torch.atan2(vecs[:, 1, 0], vecs[:, 0, 0]).unsqueeze(-1)

        # Calculate the width and height of the ellipse from eigenvalues for each matrix in the batch
        width_height = 2 * torch.sqrt(vals)

        # # inv_co_var = torch.inverse(covar + torch.eye(2, device=covar.device)*1e-6)
        vector = torch.cat([means_in_pm_1, width_height, theta, data['colors']], dim=1)
        vector = create_exclusive_patches(vector, patch_size=5)
        # vector = torch.cat([data['means'], covar.reshape(-1, 4)[:, (0, 1, 3)], data['colors']], dim=1)

        if self.normalize_gaus_params:
            vector = (torch.clip(vector, -self.feat_trunc, self.feat_trunc) - self.min_feat) \
                     / (self.max_feat - self.min_feat)
            vector = (vector - 0.5 ) * 2

        return vector, self.attr[idx]


def get_gp_celba_loaders(batch_size, return_test_loader=False, normalize_gaus_params=False):
    data_root = configs.celeba_config.datapath
    train_set = GaussianPixelDataset(root_dir=data_root, split='train', normalize_gaus_params=normalize_gaus_params)
    validation_set = GaussianPixelDataset(root_dir=data_root, split='valid', normalize_gaus_params=normalize_gaus_params)
    if return_test_loader:
        test_set = GaussianPixelDataset(root_dir=data_root, split='test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # Example usage
    data_dir = '/home/pghosh/repos/datasets/celeba/'  # Replace with your actual data directory path
    dataset = GaussianPixelDataset(root_dir=data_dir, split='train', normalize_gaus_params=False)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # Example of iterating over the DataLoader
    for vector, attr in data_loader:
        print(f'vec shape {vector.shape}, and attr shape {attr.shape}')  # Each 'vector' is a batch of the concatenated vectors
        break  # Just to demonstrate, remove this in actual use
