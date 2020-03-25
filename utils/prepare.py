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
from models.laji import lajimodel
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


def DeepTTEco(data):
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

with open('data/porto/edgeinfodict.pkl', 'rb') as f:
    porto_edgeinfo = pickle.load(f)
with open('data/porto/edgesgdict.pkl', 'rb') as f:
    porto_edgesgembed = pickle.load(f)
with open('data/chengdu/chengdu_edgeinfo.pkl', 'rb') as f:
    chengdu_edgeinfo = pickle.load(f)
with open('data/chengdu/chengdu_sgdict.pkl', 'rb') as f:
    chengdu_edgesgembed = pickle.load(f)
highway = {'living_street':1, 'morotway':2, 'motorway_link':3, 'plannned':4, 'trunk':5, "secondary":6, "trunk_link":7, "tertiary_link":8, "primary":9, "residential":10, "primary_link":11, "unclassified":12, "tertiary":13, "secondary_link":14}
bridge = {"viaduct":1, "yes":2}
tunnel = {"building_passage":1, "culvert":2, "yes":3}


def porto_TTE(data):
    scaler = StandardScaler()
    scaler.fit([[0,0]])
    scaler.mean_ = [299.31330928,2439.16007178]
    scaler.scale_ = [112.77445826, 2060.47711324]
    scaler2 = StandardScaler()
    scaler2.fit([[0,0,0,0]])
    scaler2.mean_ = [-8.62071443, 41.1573404 , -8.62074322, 41.15739367]
    scaler2.scale_ = [0.02348218, 0.0090746 , 0.02350346, 0.00906692]

    # time = torch.Tensor(scaler2.transform([[len(k[1]) * 15 - 15 for k in data]])[0])
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
            info = porto_edgeinfo[x]
            infot = []
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            # infot.append(bridge[info[5]] if info[5] in bridge.keys() else 0)
            # infot.append(tunnel[info[6]] if info[6] in tunnel.keys() else 0)
            # print(infot)
            infot.append(info[1])
            infot.append(length)
            length += info[1]
            infot += list(date)
            infot += info[7:11]
            # infot.append(info[2] if type(info[2]) == type(1.1) and not np.isnan(info[2]) else 0)
            # print(type(info[2]))
            # infot.append(info[3] if type(info[3]) == type(1.1) and not np.isnan(info[3]) else 0)
            # infot.append(info[4] if type(info[4]) == type(1.1) and not np.isnan(info[4]) else 0)
            infot += porto_edgesgembed[x] if x in porto_edgesgembed.keys() else [0 for _ in porto_edgesgembed[list(porto_edgesgembed)[0]]]

            infos.append(np.asarray(infot))
        return infos

    links = np.asarray([info(b, dateinfo[ind]) for ind, b in enumerate(links)])
    mask_dim = 42
    mask = np.arange(lens.max()) < lens[:, None]
    padded = np.zeros((*mask.shape, mask_dim), dtype=np.float32)
    con_links = np.concatenate(links)
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 6:10] = scaler2.transform(con_links[:, 6:10])
    padded[mask] = con_links
    # array = np.array([[[1],[2]],[[1],[2],[3]],[[1],[2],[3],[4],[5]]])

    # if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
    #     padded = utils.normalize(padded, key)

    padded = torch.Tensor(padded).float()
    inds = [l[0] for l in data]
    return {'links':padded, 'lens':torch.Tensor(lens).int(), 'inds':inds}, time

def chengdu_TTE(data):
    scaler = StandardScaler()
    scaler.fit([[0,0]])
    scaler.mean_ = [211.75029418, 4167.83254612]
    scaler.scale_ = [250.260086, 4257.48943]
    scaler2 = StandardScaler()
    scaler2.fit([[0,0,0,0]])
    scaler2.mean_ = [104.06487156,   30.65727836, 104.0648087 ,   30.65731526]
    scaler2.scale_ = [0.0364377245, 0.0287985467, 0.0364678498, 0.0288038976]
    time = torch.Tensor([d[-1] for d in data])
    links = []
    dateinfo = []
    for ind, l in enumerate(data):
        links.append(l[2])  # todo?
        dateinfo.append(l[3:6])
    lens = np.asarray([len(k) for k in links])

    def info(xs, date):
        # highway length lanes maxspeed width bridge tunnel
        infos = []
        length = 0
        for x in xs:
            # if x == 0:
            #     return np.asarray([0 for _ in range(7)])
            info = chengdu_edgeinfo[x]
            infot = []  # 1+2+3+4+32 # 3 + 3 + 5 + 32
            infot.append(highway[info[0]] if info[0] in highway.keys() else 0)
            # infot.append(bridge[info[5]] if info[5] in bridge.keys() else 0)
            # infot.append(tunnel[info[6]] if info[6] in tunnel.keys() else 0)

            # print(infot)
            infot.append(info[1])
            infot.append(length)  #
            length += info[1]  # todo normalize
            infot += list(date)
            infot += info[7:11]
            # lanes maxspeed width
            # infot.append(info[2] if type(info[2]) == type(1.1) and not np.isnan(info[2]) else 0)
            # print(type(info[2]))
            # infot.append(info[3] if type(info[3]) == type(1.1) and not np.isnan(info[3]) else 0)
            # infot.append(info[4] if type(info[4]) == type(1.1) and not np.isnan(info[4]) else 0)
            if x in chengdu_edgesgembed.keys():
                infot += chengdu_edgesgembed[x]
            else:
                infot+=[1 for _ in chengdu_edgesgembed[list(chengdu_edgesgembed)[0]]]
                # print(f"no sgmatch for {x}")

            infos.append(np.asarray(infot))
        return infos
    links = np.asarray([info(b, dateinfo[ind]) for ind, b in enumerate(links)])
    mask_dim = 1+2+3+4+32
    mask = np.arange(lens.max()) < lens[:, None]
    padded = np.zeros((*mask.shape, mask_dim), dtype=np.float32)
    con_links = np.concatenate(links)
    con_links[:, 1:3] = scaler.transform(con_links[:, 1:3])
    con_links[:, 6:10] = scaler2.transform(con_links[:, 6:10])
    padded[mask] = con_links
    padded = torch.Tensor(padded).float()

    inds = [l[0] for l in data]
    return {'links':padded, 'lens':torch.Tensor(lens).int(), 'inds':inds}, time


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        if isinstance(dataset[0], dict):
            self.lengths = [len(d['lats']) for d in dataset]
        else:
            self.lengths = [len(d[2]) for d in dataset]
        self.indices = list(range(self.count))

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


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
        try:
            lens = [len(d[2]) for d in tdata]
            data[phase] = tdata[np.asarray(lens) > 6]
        except:
            data[phase] = tdata
        # data[phase] = tdata

    for phase in phases:
        print(data[phase].shape)
        # loader[phase] = DataLoader(Datadict(data[phase]),
        #                            data_config['batch_size'],
        #                            collate_fn=eval(data_config['collate_fn']), shuffle=True, drop_last=True)
        loader[phase] = DataLoader(Datadict(data[phase]),
                                   batch_sampler=BatchSampler(data[phase], data_config['batch_size']),
                                   collate_fn=eval(data_config['collate_fn']))
    return loader, StandardScaler2(mean=args.data_config['time_mean'], std=args.data_config['time_std'])


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
    elif args.model == "laji":
        args.lossinside = False
        return lajimodel()


def create_loss(args):
    if args.loss == 'rmse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            rmse = torch.sqrt(torch.mean(torch.pow(preds - labels, 2)))
            return rmse
    elif args.loss == 'mse':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mse = torch.mean(torch.pow(preds - labels, 2))
            return mse
    elif args.loss == 'mape':
        def loss(**kwargs):
            preds = kwargs['predict']
            labels = kwargs['truth']
            mape = torch.mean(torch.abs(preds - labels) / (labels + 10))
            return mape
    else:
        raise ValueError("Unknown loss function.")
    return loss


if __name__ == "__main__":
    pass
