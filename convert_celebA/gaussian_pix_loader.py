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

CSV = namedtuple("CSV", ["header", "index", "data"])


class GaussianPixelDataset(Dataset):
    def __init__(self, root_dir, split, normalize_mean=False):
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
        self.normalize_mean = normalize_mean
        self.max_pos = torch.tensor([2.5512356758117676, 2.7300338745117188])
        self.min_pos = torch.tensor([-1.637588620185852, -1.1793944835662842])
        self.root = root_dir
        self.base_folder = 'celeba'
        self.gaussian_pix_dir = os.path.join(self.root, 'gps')
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
        file_path = os.path.join(self.gaussian_pix_dir, self.filenames_this_split[idx])
        data = torch.load(file_path, map_location=torch.device('cpu'))
        if self.normalize_mean:
            mean = ((data['means'] - self.min_pos)/(self.max_pos - self.min_pos) - 0.5) * 2
        else:
            mean = data['means']

        # Concatenate cvar from L_params
        L = torch.tril(data['L_params'])
        L.diagonal(dim1=-2, dim2=-1).exp_()

        covar = (L@L.transpose(1,2)).reshape(-1, 4)[:, (0, 1, 3)]
        vector = torch.cat([mean, covar, data['colors']], dim=1)

        return vector, self.attr[idx]


def get_gp_celba_loaders(batch_size, return_test_loader=False, normalize_mean=False):
    data_root = configs.celeba_config.datapath
    train_set = GaussianPixelDataset(root_dir=data_root, split='train', normalize_mean=normalize_mean)
    validation_set = GaussianPixelDataset(root_dir=data_root, split='valid', normalize_mean=normalize_mean)
    if return_test_loader:
        test_set = GaussianPixelDataset(root_dir=data_root, split='test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)
    else:
        test_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # Example usage
    data_dir = '/home/pghosh/repos/datasets/celeba/'  # Replace with your actual data directory path
    dataset = GaussianPixelDataset(root_dir=data_dir, split='train', normalize_mean=True)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # Example of iterating over the DataLoader
    for vector, attr in data_loader:
        print(f'vec shape {vector.shape}, and attr shape {attr.shape}')  # Each 'vector' is a batch of the concatenated vectors
        break  # Just to demonstrate, remove this in actual use
