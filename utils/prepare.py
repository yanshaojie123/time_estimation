import numpy as np
import os
import json

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from models.DeepTTE import DeepTTE
from utils.loss import masked_rmse_loss, masked_mse_loss

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        return torch.Tensor(self.inputs[item]).float(), torch.Tensor(self.targets[item]).float()

    def __len__(self):
        return len(self.inputs)


def default(x):
    return x


def load_dataset(args):
    absPath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(absPath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config

    data = {}
    loader = {}
    phases = ['train', 'val', 'test']

    for phase in phases:
        cat_data = np.load(os.path.join(data_config['data_dir'], phase + '.npz'))
        data['x_' + phase] = cat_data['x']
        data['y_' + phase] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for phase in phases:
        data['x_' + phase][..., 0] = scaler.transform(data['x_' + phase][..., 0])
        data['y_' + phase][..., 0] = scaler.transform(data['y_' + phase][..., 0])

    loader['scaler'] = scaler

    for phase in phases:
        print(data['x_' + phase].shape)
        loader[phase] = DataLoader(MyDataset(data['x_' + phase], data['y_' + phase]), data_config['batch_size'],
                                   collate_fn=eval(data_config['collate_fn']), shuffle=True, drop_last=True)
    return loader, scaler


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    if args.model == "deeptte":
        args.lossinside = model_config['lossinside'] == 1
        return DeepTTE(**model_config)


def create_loss(args):
    # if loss_type == 'mse_loss':
    #     return convert_to_gpu(MSELoss())
    # elif loss_type == 'bce_loss':
    #     return convert_to_gpu(BCELoss())
    if args.loss == 'masked_rmse_loss':
        return masked_rmse_loss(args.scaler, 0.0)
    elif args.loss == 'masked_mse_loss':
        return masked_mse_loss(args.scaler, 0.0)
    else:
        raise ValueError("Unknown loss function.")


if __name__ == "__main__":
    pass
