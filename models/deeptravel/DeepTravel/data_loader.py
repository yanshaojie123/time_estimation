import ast
import collections
import json
import torch
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader

# from models.deeptravel.DeepTravel import utils
import utils
import time

class MySet(Dataset):
    def __init__(self, input_file):
        # processed_data includes G_X and G_Y indicies
        # day_bin, hour_bin, time_bin (number of 5-minutes bin) sequences for each path extracted
        # sequence of driving_state vectors extracted for each cell for each path

        # print("read begin "+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        # with open('./processed_data/' + input_file, 'rb') as file:
        #     self.content = pickle.load(file)
        # print("length begin " + time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        # self.lengths = [x[1]['day_bin'].shape[1] for x in self.content]

        # input_file = 'val'
        # with open('./processed_data/' + input_file, 'r') as file:
        #     print("read start"+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        #     self.content = file.readlines()  # todo
        # print("load begin"+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        # self.content = [json.loads(x) for x in self.content]
        with open('./processed_data/' + input_file, 'rb') as file:
            print("read start"+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
            self.content = pickle.load(file)  # todo
        # with open('./processed_data/' + 'trainjson.json', 'r') as file:
        #     print("read start" + time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        #     self.content = json.load(file)['data']

        # print("length begin"+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
        # self.lengths = [len(x['G_X']) for x in self.content]

        # # with open('./traffic_features/short_ttf', 'r') as file:
        # #     self.short_ttf = file.readlines()
        # # self.short_ttf = [ast.literal_eval(x) for x in self.short_ttf]
        # #
        # # with open('./traffic_features/long_ttf', 'r') as file:
        # #     self.long_ttf = file.readlines()
        # # self.long_ttf = [ast.literal_eval(x) for x in self.long_ttf]
        # # self.short_ttf = [ast.literal_eval(x) for x in self.short_ttf]
        # with open('./traffic_features/short_ttf.pkl', 'rb') as file:
        #     self.short_ttf = pickle.load(file)
        # with open('./traffic_features/long_ttf.pkl', 'rb') as file:
        #     self.long_ttf = pickle.load(file)

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.content)


def collate_fn(data, short_ttf, long_ttf, d_n=2):

    stat_attrs = ['dist', 'time']
    temporal_f = ['day_bin', 'hour_bin', 'time_bin']
    spatial_f = ['G_X', 'G_Y']
    driving_state = ['dr_state']

    helpers_k = ['time_gap', 'borders', 'mask']

    stats, temporal, spatial = {}, {}, {}

    lens = np.asarray([len(item['G_X']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        stats[key] = utils.normalize(x, key)

    for key in temporal_f:
        seqs = np.asarray([item[key] for item in data])
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)

        padded = torch.from_numpy(padded).long()
        temporal[key] = padded

    for key in spatial_f:
        seqs = np.asarray([item[key] for item in data])
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate(seqs)

        padded = torch.from_numpy(padded).long()
        spatial[key] = padded

    for key in driving_state:
        seqs = np.asarray([item[key] for item in data])
        # as each driving_state vector has length 4
        mask = np.arange(lens.max() * 4) < lens[:, None] * 4
        padded = np.zeros(mask.shape, dtype=np.float32)
        padded[mask] = np.concatenate([np.concatenate(item) for item in seqs])
        dr_state = torch.from_numpy(padded)

    lens = lens.tolist()
    spatial['lens'] = lens

    short, long = [[] for _ in data], [[] for _ in data]

    for ind, item in enumerate(data):
        # extracting short (for specified number of neighbour layers) and long-term features for each cell in the path
        # aggregating short term feature extracted from different cells of the same layer
        for x_ind, y_ind, time_bin in zip(item['G_X'], item['G_Y'], item['time_bin']):
            ext_short, ext_long = extract_ttf_for_cell(x_ind, y_ind, short_ttf, d_n, time_bin, long_ttf)
            short[ind].append(ext_short)
            long[ind].append(ext_long)

    fulfill_missed_ttf(short, long, temporal['time_bin'], temporal['day_bin'])
    short, long = form_ttf_vectors(short, long, temporal['time_bin'], temporal['day_bin'])

    helpers = {}
    for key in helpers_k:
        helpers[key] = data[0][key]

    return stats, temporal, spatial, dr_state, short, long, helpers


def extract_ttf_for_cell(x_ind, y_ind, short, d_n, time_bin, long,  min_cl=12):
    """
    Aggregate short-term traffic features for one cell.
    Use only close bins features (min_cl (closeness) is the biggest distance between time-bins)
    Aggregate traffic features for each neighbour layer.

    As the result:
    {0 (layer): {time_bin: {features (av_speed, av_time, n)}, time_bin: {features},

     1: {time_bin: {features}, time_bin: {features},
     ...
     n: {time_bin: {features}, time_bin: {features}
    }
    """
    short_ttf = {0: {}}
    long_ttf = {}

    for bin, values in short[x_ind][y_ind].items():
        bin = int(bin)
        if check_closeness(time_bin, bin):
            short_ttf[0][bin] = short[x_ind][y_ind][str(bin)]

    for d in range(1, d_n + 1):
        short_ttf[d] = []
        try:
            for y in range((y_ind - d), (y_ind + d)):
                if y == (x_ind - d) or y == (y_ind + d):
                    for x in range((x_ind - d), (x_ind + d)):
                        if short[x][y]:
                            short_ttf[d].append(short[x][y])
                else:
                    if short[x_ind - d][y]:
                        short_ttf[d].append(short[x_ind - d][y])
                    if short[x_ind + d][y]:
                        short_ttf[d].append(short[x_ind + d][y])
        except IndexError:
            pass
    if long[x_ind][y_ind]:
        long_ttf = {int(day): features for day, features in long[x_ind][y_ind].items()}

    aggregate_short_ttf(short_ttf, time_bin)

    return short_ttf, long_ttf


def aggregate_short_ttf(short, time_bin, min_cl=12):
    """Calculate average speed and time for time_bin according to traffic features from all cells."""
    for d_n in range(1, max(short.keys()) + 1):
        if len(short[d_n]) > 1:
            close_bins = collections.defaultdict(lambda: {'speed_sum': 0.0, 'time_sum': 0.0, 'n_sum': 0})
            for cell_ttf in short[d_n]:
                for bin, values in cell_ttf.items():
                    bin = int(bin)
                    if check_closeness(time_bin, bin, min_cl):
                        close_bins[bin]['speed_sum'] += values['speed'] * values['n']
                        close_bins[bin]['time_sum'] += values['time'] * values['n']
                        close_bins[bin]['n_sum'] += values['n']

            for bin, values in close_bins.items():
                close_bins[bin]['speed'] = values['speed_sum'] / values['n_sum']
                close_bins[bin]['time'] = values['time_sum'] / values['n_sum']
                close_bins[bin]['n'] = values['n_sum']
                del close_bins[bin]['speed_sum']
                del close_bins[bin]['time_sum']
                del close_bins[bin]['n_sum']

            short[d_n] = dict(close_bins)
        else:
            short[d_n] = dict() if len(short[d_n]) == 0 else dict(short[d_n][0])
            short[d_n] = {int(k): v for k, v in short[d_n].items()}
            short[d_n] = {k: v for k, v in short[d_n].items() if check_closeness(time_bin, k, min_cl)}
    return


def check_closeness(a_bin, b_bin, min_cl=12):
    return (a_bin >= b_bin and a_bin - b_bin <= min_cl) or (a_bin < b_bin and a_bin + 287 - b_bin <= min_cl)


def form_ttf_vectors(short_ttf, long_ttf, time_bins, day_bins):
    """For short ttf:
            Collect traffic feature vectors for each separate layer in order closeness decreasing from 12 to 0 (current)
       Long ttf:
            Collect long traffic feature vectors in order closeness decreasing from 6 to 0 (current)
    """
    new_short = []

    for ttf, cell_tbin in zip(short_ttf[0], time_bins[0]):
        new_short.append([])
        for layer in ttf.keys():
            short_vectors = []
            for tbin in sorted(ttf[layer]):
                # generate vector R4 for not empy time-bins for the cell
                short_vectors.append([
                    int(cell_tbin) - tbin,
                    ttf[layer][tbin]['speed'], ttf[layer][tbin]['time'],
                    ttf[layer][tbin]['n']
                ])

            new_short[-1].append(torch.Tensor(short_vectors))

    new_long = []
    for ttf, cell_day in zip(long_ttf[0], day_bins[0]):
        long_vectors = []
        for day, features in ttf.items():
            if int(cell_day) >= day:
                j_cl = int(cell_day) - day
            else:
                j_cl = int(cell_day) - day + 7
            long_vectors.append([j_cl, ttf[day]['speed'], ttf[day]['time'], ttf[day]['n']])
        new_long.append(torch.Tensor(long_vectors))

    return new_short, new_long


def fulfill_missed_ttf(short, long, time_bins, day_bins):

    for short_path, long_path, time_bins_path, day_bin_path in zip(short, long, time_bins, day_bins):
        for short_cell, long_cell, time_bin, day_bin in zip(short_path, long_path, time_bins_path, day_bin_path):
            if time_bin - 12 >= 0:
                close_time_bins = list(range(time_bin - 12, time_bin + 1))
            else:
                close_time_bins = list(range(0, time_bin + 1)) + list(range(time_bin - 12 + time_bin, 280))
            for layer in range(max(short_cell.keys()) + 1):
                # for time_cl in close_time_bins:
                #     if not short_cell[layer].get(time_cl):
                #         short_cell[layer][time_cl] = {'speed': 0, 'time': 0, 'n': 0}
                if not short_cell[layer].get(time_bin):
                    short_cell[layer][int(time_bin)] = {'speed': 27.238, 'time': 35.56, 'n': 1}

            # for day_cl in range(0, 7):
            #     if not long_cell.get(day_cl):
            #         long_cell[day_cl] = {'speed': 0, 'time': 0, 'n': 0}
            if not long_cell.get(int(0)):
                long_cell[0] = {'speed': 27.238, 'time': 35.56, 'n': 1}


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
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
            partial_indices.sort(key=lambda x: self.lengths[x], reverse=True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def myco(data):
    return data[0]

def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)
    print("opendataset")

    # batch_sampler = BatchSampler(dataset, batch_size)
    # print("getbatchsampler")
    # data_loader = DataLoader(
    #     dataset = dataset,
    #     collate_fn = lambda x: collate_fn(x, dataset.short_ttf, dataset.long_ttf),
    #     # num_workers = 1, # set to one for easy debugging
    #     batch_sampler = batch_sampler,
    #     # pin_memory = True
    #
    # )
    data_loader = DataLoader(
        dataset = dataset,

        # collate_fn = lambda x: collate_fn(x, dataset.short_ttf, dataset.long_ttf),
        # num_workers = 1, # set to one for easy debugging
        # batch_sampler = batch_sampler,
        # pin_memory = True

        collate_fn=myco,
        batch_size=batch_size,
        shuffle=True
    )

    print("getdataloader")
    return data_loader
