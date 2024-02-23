import os
import csv
import torch
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader

CSV = namedtuple("CSV", ["header", "index", "data"])


class GaussianPixelDataset(Dataset):
    def __init__(self, root_dir, split):
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

        self.filenames_this_split = []
        self.attr = []
        for i, curr_file in enumerate(splits.index):
            if mask != slice(None) and not mask[i]:
                continue
            gp_rep = f'{os.path.splitext(curr_file)[0]}.pt'
            if gp_rep in self.all_file_names:
                self.filenames_this_split.append(gp_rep)
                self.attr.append(attr.data[i])

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

        # Concatenate cvar from L_params
        L = torch.tril(data['L_params'])
        L.diagonal(dim1=-2, dim2=-1).exp_()

        covar = (L@L.transpose(1,2)).reshape(-1, 4)[:, (0, 1, 3)]
        vector = torch.cat([data['means'], covar, data['colors']], dim=1)

        return vector, self.attr[idx]


if __name__ == '__main__':
    # Example usage
    data_dir = '/home/pghosh/repos/datasets/celeba/'  # Replace with your actual data directory path
    dataset = GaussianPixelDataset(root_dir=data_dir, split='train')
    data_loader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=4)

    # Example of iterating over the DataLoader
    for vector, attr in data_loader:
        print(f'vec shape {vector.shape}, and attr shape {attr.shape}')  # Each 'vector' is a batch of the concatenated vectors
        break  # Just to demonstrate, remove this in actual use
