import numpy as np
import os
import json
import pandas as pd
import pickle

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from models.DeepTTE import DeepTTE
from models.TTEModel import TTEModel
from models.TTEModelo import TTEModel as TTEModelo
from utils.loss import masked_rmse_loss, masked_mse_loss, masked_mape_loss
from sklearn.preprocessing import StandardScaler


class StandardScaler2:
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


# class MyDataset(Dataset):
#     def __init__(self, inputs, targets):
#         self.inputs = inputs
#         self.targets = targets
#
#     def __getitem__(self, item):
#         return torch.Tensor(self.inputs[item]).float(), torch.Tensor(self.targets[item]).float()
#
#     def __len__(self):
#         return len(self.inputs)

class Datadict(Dataset):
    def __init__(self, inputs):
        self.content = inputs

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)
        # return 100

def default(x):
    return x


def cdtte(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        # attr[key] = utils.normalize(x, key)
        attr[key] = x

    for key in info_attrs:
        if key == "driverID":
            attr[key] = torch.LongTensor([item[key]%24000 for item in data])
            # todo
        else:
            attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        seqs = np.asarray([item[key] for item in data])
        # print([len(item[key]) for item in data])
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)

        # if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
        #     padded = utils.normalize(padded, key)

        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return {'attr': attr, 'traj': traj}, attr['time']


# def load_dataset(args):
#     absPath = os.path.join(os.path.dirname(__file__), "data_config.json")
#     with open(absPath) as file:
#         data_config = json.load(file)[args.dataset]
#         args.data_config = data_config
#
#     data = {}
#     loader = {}
#     phases = ['train', 'val', 'test']
#
#     for phase in phases:
#         cat_data = np.load(os.path.join(data_config['data_dir'], phase + '.npz'))
#         data['x_' + phase] = cat_data['x']
#         data['y_' + phase] = cat_data['y']
#     scaler = StandardScaler2(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
#     # Data format
#     for phase in phases:
#         data['x_' + phase][..., 0] = scaler.transform(data['x_' + phase][..., 0])
#         data['y_' + phase][..., 0] = scaler.transform(data['y_' + phase][..., 0])
#
#     loader['scaler'] = scaler
#
#     for phase in phases:
#         print(data['x_' + phase].shape)
#         loader[phase] = DataLoader(MyDataset(data['x_' + phase], data['y_' + phase]), data_config['batch_size'],
#                                    collate_fn=eval(data_config['collate_fn']), shuffle=True, drop_last=True)
#     return loader, scaler


with open('data/porto/edgeinfodict.pkl', 'rb') as f:
    porto_edgeinfo = pickle.load(f)
with open('data/porto/edgesgdict.pkl', 'rb') as f:
    porto_edgesgembed = pickle.load(f)
with open('data/chengdu/chengdu_edgeinfo.pkl', 'rb') as f:
    chengdu_edgeinfo = pickle.load(f)
with open('data/chengdu/chengdu_sgdict.pkl', 'rb') as f:
    chengdu_edgesgembed = pickle.load(f)
def portoedge(data):
    # with open('data/porto/edgeinfodict.pkl', 'rb') as f:
    #     edgeinfo = pickle.load(f)
    # with open('data/porto/edgesgdict.pkl', 'rb') as f:
    #     edgesgembed = pickle.load(f)
    highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
    bridge = {"viaduct":1, "yes":2}
    tunnel = {"building_passage":1, "culvert":2, "yes":3}
    scaler = StandardScaler()
    scaler.fit([[0,0,0,0,0]])
    # scaler.mean_ = [104.09531247, 0., 0., 0.]
    # scaler.scale_ = [131.50032485, 1., 1., 1.]
    scaler2 = StandardScaler()
    scaler2.fit([[0]])
    # scaler2.mean_ = [490.5749094979864]
    # scaler2.scale_ = [231.25910758152915]

    # time = torch.Tensor(scaler2.transform([[len(str(k[2]).split(',')) * 15 -15 for k in data]])[0])
    time = torch.Tensor(scaler2.transform([[len(k[1]) * 15 - 15 for k in data]])[0])

    links = []
    dateinfo = []
    for ind, l in enumerate(data):
        # if type(l[3])!=type(' '):  # todo
        #     links.append([0])
        #     time[ind] = 0
        #     # print(type(l[3]))
        # else:
        #     links.append([int(num) for num in str(l[3]).split(',')])  # todo
        links.append(l[2])  # todo
        dateinfo.append(l[3:])
    lens = np.asarray([len(k) for k in links])
    def info(xs, date):
        # highway length lanes maxspeed width bridge tunnel
        infos = []
        length = 0
        for x in xs:
            if x == 0:
                return np.asarray([0 for _ in range(7)])
            info = porto_edgeinfo[x]
            infot = []  # 3 + 3 + 5 + 32
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(bridge[info[5]] if info[5] in bridge.keys() else 0)
            infot.append(tunnel[info[6]] if info[6] in tunnel.keys() else 0)
            infot += list(date)
            # print(infot)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            infot.append(info[2] if type(info[2]) == type(1.1) and not np.isnan(info[2]) else 0)
            # print(type(info[2]))
            infot.append(info[3] if type(info[3]) == type(1.1) and not np.isnan(info[3]) else 0)
            infot.append(info[4] if type(info[4]) == type(1.1) and not np.isnan(info[4]) else 0)
            infot += porto_edgesgembed[x] if x in porto_edgesgembed.keys() else [0 for _ in porto_edgesgembed[list(porto_edgesgembed)[0]]]

            infos.append(np.asarray(infot))
        return infos

    links = np.asarray([info(b, dateinfo[ind]) for ind, b in enumerate(links)])
    mask_dim = 3 + 3 + 5 + 32
    mask = np.arange(lens.max()) < lens[:, None]
    padded = np.zeros((*mask.shape, mask_dim), dtype=np.float32)
    con_links = np.concatenate(links)
    con_links[:, 6:11] = scaler.transform(con_links[:, 6:11])
    padded[mask] = con_links
    # array = np.array([[[1],[2]],[[1],[2],[3]],[[1],[2],[3],[4],[5]]])

    # if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
    #     padded = utils.normalize(padded, key)

    padded = torch.Tensor(padded).float()
    inds = [l[0] for l in data]
    return {'links':padded, 'lens':torch.Tensor(lens).int(), 'inds':inds}, time

# todo unify

def chengduTTE(data):
    highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
    bridge = {"viaduct":1, "yes":2}
    tunnel = {"building_passage":1, "culvert":2, "yes":3}
    scaler = StandardScaler()
    scaler.fit([[0,0,0,0,0]])
    scaler2 = StandardScaler()
    scaler2.fit([[0]])
    time = torch.Tensor([d[-1] for d in data])

    links = []
    dateinfo = []
    for ind, l in enumerate(data):
        links.append(l[2])  # todo
        dateinfo.append(l[3:6])
    lens = np.asarray([len(k) for k in links])
    def info(xs, date):
        # highway length lanes maxspeed width bridge tunnel
        infos = []
        length = 0
        for x in xs:
            if x == 0:
                return np.asarray([0 for _ in range(7)])
            info = chengdu_edgeinfo[x]
            infot = []  # 3 + 3 + 5 + 32
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            infot.append(bridge[info[5]] if info[5] in bridge.keys() else 0)
            infot.append(tunnel[info[6]] if info[6] in tunnel.keys() else 0)
            infot += list(date)
            # print(infot)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            infot.append(info[2] if type(info[2]) == type(1.1) and not np.isnan(info[2]) else 0)
            # print(type(info[2]))
            infot.append(info[3] if type(info[3]) == type(1.1) and not np.isnan(info[3]) else 0)
            infot.append(info[4] if type(info[4]) == type(1.1) and not np.isnan(info[4]) else 0)
            infot += chengdu_edgesgembed[x] if x in chengdu_edgesgembed.keys() else [0 for _ in chengdu_edgesgembed[list(chengdu_edgesgembed)[0]]]

            infos.append(np.asarray(infot))
        return infos

    links = np.asarray([info(b, dateinfo[ind]) for ind, b in enumerate(links)])
    mask_dim = 3 + 3 + 5 + 32
    mask = np.arange(lens.max()) < lens[:, None]
    padded = np.zeros((*mask.shape, mask_dim), dtype=np.float32)
    con_links = np.concatenate(links)
    con_links[:, 6:11] = scaler.transform(con_links[:, 6:11])
    padded[mask] = con_links

    padded = torch.Tensor(padded).float()
    inds = [l[0] for l in data]
    return {'links':padded, 'lens':torch.Tensor(lens).int(), 'inds':inds}, time

def load_datadict(args):
    abspath = os.path.join(os.path.dirname(__file__), "data_config.json")
    with open(abspath) as file:
        data_config = json.load(file)[args.dataset]
        args.data_config = data_config

    data = {}
    loader = {}
    if args.mode == 'test':
        phases = ['test']
    else:
        phases = ['train', 'val', 'test']

    for phase in phases:
        tdata = np.load(os.path.join(data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        lens = [len(d[2]) for d in tdata]
        data[phase] = tdata[np.asarray(lens) > 2]
        # tdata = np.load(os.path.join(data_config['data_dir'], phase + '.npy'), allow_pickle=True)
        # data[phase] = tdata

    for phase in phases:
        print(data[phase].shape)
        loader[phase] = DataLoader(Datadict(data[phase]), data_config['batch_size'],
                                   collate_fn=eval(data_config['collate_fn']), shuffle=True, drop_last=True)
    return loader, StandardScaler2(mean=0, std=1)
    # return loader, StandardScaler2(mean=490.5749094979864, std=231.25910758152915)


def create_model(args):
    absPath = os.path.join(os.path.dirname(__file__), "model_config.json")
    with open(absPath) as file:
        model_config = json.load(file)[args.model]
    args.model_config = model_config
    if args.model == "DeepTTE":
        args.lossinside = model_config['lossinside'] == 1
        return DeepTTE(**model_config)
    elif args.model == "TTEModel":
        args.lossinside = False
        return TTEModel(**model_config)
    elif args.model == "TTEModelo":
        args.lossinside = False
        return TTEModelo(**model_config)


def create_loss(args):
    if args.loss == 'masked_rmse_loss':
        return masked_rmse_loss(args.scaler, 0.0)
    elif args.loss == 'masked_mse_loss':
        return masked_mse_loss(args.scaler, 0.0)
    elif args.loss == 'masked_mape_loss':
        return masked_mape_loss(args.scaler, 0.0)
    else:
        raise ValueError("Unknown loss function.")


if __name__ == "__main__":
    pass
