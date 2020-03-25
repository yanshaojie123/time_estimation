import json
import numpy as np
import time

def deeptravelinrec():
    with open('../../../data/porto/deeptravel/traino') as dfile:  # todo
        data = dfile.readlines()
        data = [json.loads(s) for s in data]
    id = []
    res = []
    for i, k in enumerate(data):
        _a = np.min(k['lats'])
        _b = np.min(k['lngs'])
        _c = np.max(k['lats'])
        _d = np.max(k['lngs'])
        if _a < 41.1356 or _b < -8.7068 or _c > 41.1922 or _d > -8.5361 or k['dist']<=0:
            id.append(i)
        else:
            res.append(k)
    with open('../../../data/porto/deeptravel/train', 'w') as f:
        for d in res:
            json.dump(d, f)
            f.write('\n')

def computedeeptravelmeanstd():
    with open('models/deeptravel/processed_data/train.txt', 'r') as f:
        data = f.readlines()
    import numpy as np
    import json
    data = [json.loads(d) for d in data]
    data = np.load("train.npy", allow_pickle=True)
    for k in ['dist_gap', 'time_gap', 'lngs',  'lats']:
        d = np.concatenate([da[k] for da in data])
        print(f'"{k}_mean": {np.mean(d)},')
        print(f'"{k}_std": {np.std(d)},')
    for k in ['dist', 'time']:
        d = [da[k] for da in data]
        print(f'"{k}_mean": {np.mean(d)},')
        print(f'"{k}_std": {np.std(d)},')
    # with open('models/deeptravel/DeepTravel/processed_data/val', 'r') as f:
    #     data = f.readlines()
    # for i in range(8):
    #     with open(f'models/deeptravel/DeepTravel/processed_data/train_{i}', 'w') as f:
    #         for d in data[int(len(data) * 0.1 * i):int(len(data) * 0.1 * (i + 1))]:
    #             f.write(d)
    # for i in range(8):
    #     print(f"train_{i},")

def lajideeptrvel():
    import os
    # os.chdir('models/deeptravel/DeepTravel')
    from models.deeptravel.DeepTravel.data_loader import get_loadero
    import pickle
    import json

    import gc
    # with open('./traffic_features/short_ttf.pkl', 'rb') as file:
    #     short_ttf = pickle.load(file)
    # with open('./traffic_features/long_ttf.pkl', 'rb') as file:
    #     long_ttf = pickle.load(file)
    #
    # data = 'val'
    # with open(f'./processed_data/{data}', 'r') as file:
    #     print("read start")
    #     contento = file.readlines()  # todo
    # for sp in range(10):
    #     print(f"========================={sp}==================")
    #     content = contento[int(len(contento)*0.1*sp):int(len(contento)*0.1*(sp+1))]
    #     content = [json.loads(x) for x in content]
    #     lengths = [len(x['G_X']) for x in content]
    #     res = []
    #     for i, d in enumerate(content):
    #         try:
    #             res.append(collate_fn([d], short_ttf, long_ttf))
    #         except Exception as e:
    #             print(e)
    #         if i % 1000 == 0:
    #             print(i)
    #     with open(f'./processed_data/{data}_{sp}.pkl', 'wb') as f:
    #         pickle.dump(res, f)
    #     del content
    #     del res
    #     del f
    #     del lengths
    #     gc.collect()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    filename = "test"
    loader = get_loadero(filename, 128)
    res = []
    for idx, data in enumerate(loader):
        if idx % 100 == 0 and idx > 0:
            with open(f'./models/deeptravel/DeepTravel/processed_data/aftercollate/{filename}_{idx // 100}.pkl', 'wb') as f:
                pickle.dump(res, f)
            print(f"save{idx // 100} at {time.strftime('%H:%M:%S'.format(time.localtime(time.time())))}")
            res.clear()
        res.append(data)