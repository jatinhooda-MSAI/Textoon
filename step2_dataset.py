import json
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

#Build a CartoonDataset that reads from your HDF5, joins to the attribute columns from metadata.csv, 
#and returns (image_tensor, attribute_vector). Then write a tiny script that pulls 16 random samples 
#and saves them as a grid with their active attributes printed below each. You eyeball that grid before writing a 
#single line of model code. 

class CartoonDataset(Dataset):
    def __init__(self, h5_path, metadata_csv, attr_cols_json, hflip=True):
        self.h5_path = str(h5_path)
        self._h5 = None

        self.attr_cols = json.load(open(attr_cols_json))
        df = pd.read_csv(metadata_csv)

        with h5py.File(self.h5_path, "r") as h5:
            h5_filenames = [fn.decode() for fn in h5["filenames"][:]]
        fn_to_idx = {fn: i for i, fn in enumerate(h5_filenames)}

        self.h5_indices = df["filename"].map(fn_to_idx).to_numpy(np.int64)
        self.attrs = df[self.attr_cols].to_numpy(np.float32)
        self.filenames = df["filename"].tolist()
        self.hflip = hflip

    def __len__(self):
        return len(self.h5_indices)

    def __getitem__(self, idx):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

        img = self._h5["images"][int(self.h5_indices[idx])]
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 127.5 - 1.0

        if self.hflip and torch.rand(1).item() < 0.5:
            img = torch.flip(img, dims=[2])

        return img, torch.from_numpy(self.attrs[idx])