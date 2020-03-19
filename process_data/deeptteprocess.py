import numpy as np
from math import radians, cos, sin, asin, sqrt
import random

def geo_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    lon1, lat1, lon2, lat2 = map(radians, map(float, [lon1, lat1, lon2, lat2]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r
def resamplefordeeptte():
    name = "val"
    data = np.load(f'deeptteb/{name}.npy', allow_pickle=True)
    for i in range(len(data)):
        longs = data[i]['lngs']
        lats = data[i]['lats']
        times = data[i]['time_gap']
        states = data[i]['states']
        dists = [geo_distance(longs[i], lats[i], longs[i+1], lats[i+1]) for i in range(len(longs)-1)]
        reduce = np.cumsum(dists)
        r = 0
        new_lngs = [longs[0]]
        new_lats = [lats[0]]
        new_dists = [0.0]
        new_times = [times[0]]
        new_states = [states[0]]
        # if name == "train":
        t = random.random() * 0.2 + 0.2
        # else:
        # t = 0.3
        for j, d in enumerate(dists):
            r += dists[j]
            if r >= t or j == len(dists)-1:
                r = 0
                new_lngs.append(longs[j+1])
                new_lats.append(lats[j+1])
                new_dists.append(reduce[j])
                new_times.append(times[j+1])
                new_states.append(states[j+1])
        data[i]['lngs'] = new_lngs
        data[i]['lats'] = new_lats
        data[i]['dist_gap'] = new_dists
        data[i]['time_gap'] = new_times
        data[i]['states'] = new_states
        data[i]['time'] = new_times[-1]
        data[i]['dist'] = new_dists[-1]
    lens = [len(d['lats']) for d in data]
    times = [d['time'] for d in data]
    np.save(f"deeptte/{name}.npy", data[np.asarray(lens) > 2])
