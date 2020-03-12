import pandas as pd
import numpy as np

def edgemap():
    edgeinfo = pd.read_csv('data/porto/fmm/original_edges.csv')
    edgemap = pd.read_csv('data/porto/fmm/edgemap.csv')  # todo oid
    edgemap = edgemap.drop(['geom', 'source', 'target'], axis=1)
    result = pd.merge(edgemap, edgeinfo, left_on='oid', right_on='gid').sort_values(by='gid_x')
    result.to_csv('data/porto/edgeinfo.csv')

def edgeinfo():
    edgeinfo = pd.read_csv('data/porto/fmm/edgeinfo.csv')
    edgedata = edgeinfo.values
    result = {}
    for k in edgedata:
        result[k[1]] = (k[6], k[8], k[12], k[14], k[16], k[17], k[18])
    with open('data/porto/edgeinfo.pkl', 'wb') as f:
        import pickle
        pickle.dump(result, f)

def checkdup():
    def duppath(x):
        if type(x) == type(" "):
            k = {}
            links = x.split(',')
            for i in range(len(links)-1, -1, -1):
                link = links[i]
                if link not in k.keys():
                    k[link] = i
                else:
                    links = links[:i] + links[k[link]:]
                    return duppath(','.join(links))
        return x
    def depmgeom(x):
        if type(x) == type(" "):
            x.replace('LINESTRING(', '')
            x.replace(')', '')
            try:
                return 'LINESTRING(' + duppath(x) + ')'
            except:
                print(len(x.split(',')))
                return 'LINESTRING(' + x + ')'
        return x
    data = pd.read_csv('data/porto/fmm/porto_train.txt', sep=';')
    data.cpath = data.cpath.map(duppath)
    data.mgeom = data.mgeom.map(depmgeom)
    data.to_csv('data/porto/train_drop.csv', sep=';')

def csvtonpy():
    data = pd.read_csv('data/porto/train_dropdup.csv', sep=';').drop('length', axis=1).values
    data = data[list(map(lambda x: type(x) == type(" "), data[:, 3]))]  # drop nan
    out = 0
    ind = []
    # drop losing path
    for i, d in enumerate(data):
        o = set(d[2].split(','))
        c = d[3].split(',')
        a = True
        for j in o:
            if j not in c:
                a = False
                out += 1
                break
        if a:
            ind.append(i)
    data = data[ind]
    train_res = np.asarray(data)
    l2 = train_res[:, 2]
    l2 = [[int(k) for k in l.split(',')] for l in l2]
    train_res[:, 2] = l2
    l2 = train_res[:, 3]
    l2 = [[int(k) for k in l.split(',')] for l in l2]
    train_res[:, 3] = l2
    data = train_res[1:]

    ratio = np.asarray([len(d[2])/len(d[1]) for d in data])
    ratioforind = np.asarray([len(d[2])/len(d[1]) for d in data])
    ratio.sort()
    low = ratio[int(ratio.shape[0]*0.05)]
    high = ratio[int(ratio.shape[0]*0.95)]
    ind = np.logical_and(ratioforind>low , ratioforind<high)
    data = data[ind]


    train_data = data[:int(len(data) * 0.8)]
    np.save('data/porto/portotrain/train.npy', train_data)
    val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    # val_data = data[data.id >= len(data) * 0.8][data.id < len(data)*0.9].drop('length', axis=1).values
    np.save('data/porto/portotrain/val.npy', val_data)
    test_data = data[int(len(data) * 0.9):]
    np.save('data/porto/portotrain/test.npy', test_data)

    # for indexp, path in enumerate(cpath):
    #     if type(path) == type("  "):
    #         k = {}
    #         for indext, link in enumerate(path.split(',')):
    #             if link not in k.keys():
    #                 k[link] = indext
    #             else:
    #                 dup_ind.append(indexp)
    #                 lens.append(indext - k[link])
    #                 break

def get_deeptravel():
    import json
    newtrain = np.load('data/porto/deeptteori/train.npy')  # del length 1 2
    t1 = np.load('data/porto/portotrain/train.npy')
    t2 = np.load('data/porto/portotrain/val.npy')
    t3 = np.load('data/porto/portotrain/test.npy')

    newtime = {}
    for i in t1:
        newtime[i[0]] = [newtrain[i[0] - 1]['weekID'], newtrain[i[0] - 1]['dateID'], newtrain[i[0] - 1]['timeID']]
    for i in t2:
        newtime[i[0]] = [newtrain[i[0] - 1]['weekID'], newtrain[i[0] - 1]['dateID'], newtrain[i[0] - 1]['timeID']]
    for i in t3:
        newtime[i[0]] = [newtrain[i[0] - 1]['weekID'], newtrain[i[0] - 1]['dateID'], newtrain[i[0] - 1]['timeID']]
    with open('data/porto/portotrain/timedict.pkl', 'wb') as f:
        import pickle
        pickle.dump(newtime, f)

    with open('data/porto/deeptravel/train', 'w') as f:
        for d in newtrain[[int(k) - 1 for k in t1[:, 0]]]:
            json.dump(d, f)
            f.write('\n')
    with open('data/porto/deeptravel/val', 'w') as f:
        for d in newtrain[[int(k) - 1 for k in t2[:, 0]]]:
            json.dump(d, f)
            f.write('\n')
    with open('data/porto/deeptravel/test', 'w') as f:
        for d in newtrain[[int(k) - 1 for k in t3[:, 0]]]:
            json.dump(d, f)
            f.write('\n')
    res = []
    for i in t1:
        res.append(newtrain[i[0] - 1])
    np.save('data/porto/deeptte/train.npy', res)
    res = []
    for i in t2:
        res.append(newtrain[i[0] - 1])
    np.save('data/porto/deeptte/val.npy', res)
    res = []
    for i in t3:
        res.append(newtrain[i[0] - 1])
    np.save('data/porto/deeptte/test.npy', res)

def addtime():
    dirt = 'data/porto/portotrain/'
    with open(f'{dirt}timedict.pkl', 'rb') as f:
        timed = pickle.load(f)
    train_data = np.load(f'{dirt}train.npy'), 'train'
    val_data = np.load(f'{dirt}val.npy'), 'val'
    test_data = np.load(f'{dirt}test.npy'), 'test'
    for data, name in [train_data, val_data, test_data]:
        time = []
        for i in data:
            time.append(timed[i[1]])
        train_res = np.concatenate([data, np.asarray(time)], axis=-1)
        l2 = train_res[:, 2]
        l2 = [[int(k) for k in l.split(',')] for l in l2]
        train_res[:, 2] = l2
        l2 = train_res[:, 3]
        l2 = [[int(k) for k in l.split(',')] for l in l2]
        train_res[:, 3] = l2
        train_res = train_res[:, [1, 2, 3, 5, 6, 7]]
        np.save(f'{dirt}{name}.npy', train_res)

def badcase():
    data = np.load('data/porto/portotrain/test.npy')
    mapd = {data[i][0]: i for i in range(47785)}
    bad = np.load('data/porto/badcase.npy').tolist()
    print(data[mapd[1275177]])

def deeptraveldelout():
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

def handldchengdu(day):
    # import time
    # import pandas as pd
    # import numpy as np
    # day = 20140803
    data = pd.read_csv(f'data/chengdu/{day}_train.txt', header=None, names=["taxiid", "lat", "lng", "state", "time"])#.iloc[:10000]
    data['timestamp'] = data['time'].apply(lambda x: int(time.mktime(time.strptime(x, "%Y/%m/%d %H:%M:%S"))))
    ids = pd.unique(data.taxiid)
    res = []
    pad = np.asarray([[0,0,0,0,0,0]])
    for id in ids:
        iddata = data[data.taxiid == id].sort_values(by='timestamp').values
        iddata = np.concatenate([iddata, pad], axis=0)
        taxiid = id
        lats = []
        lngs = []
        time_gap = []
        start = 0

        for k in iddata:
            if k[3] == 1:
                if len(lats) == 0:
                    start = k[5]
                    lats.append(k[1])
                    lngs.append(k[2])
                    time_gap.append(k[5] - start)
                else:
                    lats.append(k[1])
                    lngs.append(k[2])
                    time_gap.append(k[5]-start)
            else:
                if len(lats) != 0:
                    res.append([taxiid, start, lats.copy(), lngs.copy(), time_gap.copy()])
                    lngs.clear()
                    lats.clear()
                    time_gap.clear()
    np.save(f"data/chengdu/npy/{day}.npy", np.asarray(res, dtype=object))
    # for d in range(20140810, 20140830):
    #     print(f"start {d} {time.strftime('%H:%M:%S',time.localtime(time.time()))}")
    #     try:
    #         handldchengdu(d)
    #     except Exception as e:
    #         print(f"error on {d} {time.strftime('%H:%M:%S',time.localtime(time.time()))}")
    #         print(e)
    #     print(f"end {d} {time.strftime('%H:%M:%S',time.localtime(time.time()))}")
def _transformlat(lng, lat):
  ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
     0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
  ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
      math.sin(2.0 * lng * pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lat * pi) + 40.0 *
      math.sin(lat / 3.0 * pi)) * 2.0 / 3.0
  ret += (160.0 * math.sin(lat / 12.0 * pi) + 320 *
      math.sin(lat * pi / 30.0)) * 2.0 / 3.0
  return ret
def _transformlng(lng, lat):
  ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
     0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
  ret += (20.0 * math.sin(6.0 * lng * pi) + 20.0 *
      math.sin(2.0 * lng * pi)) * 2.0 / 3.0
  ret += (20.0 * math.sin(lng * pi) + 40.0 *
      math.sin(lng / 3.0 * pi)) * 2.0 / 3.0
  ret += (150.0 * math.sin(lng / 12.0 * pi) + 300.0 *
      math.sin(lng / 30.0 * pi)) * 2.0 / 3.0
  return ret
def gcj02_to_wgs84(lng, lat):
  """
  GCJ02(火星坐标系)转GPS84
  :param lng:火星坐标系的经度
  :param lat:火星坐标系纬度
  :return:
  """
  # import math
  x_pi = 3.14159265358979324 * 3000.0 / 180.0
  pi = 3.1415926535897932384626  # π
  a = 6378245.0  # 长半轴
  ee = 0.00669342162296594323  # 扁率
  dlat = _transformlat(lng - 105.0, lat - 35.0)
  dlng = _transformlng(lng - 105.0, lat - 35.0)
  radlat = lat / 180.0 * pi
  magic = math.sin(radlat)
  magic = 1 - ee * magic * magic
  sqrtmagic = math.sqrt(magic)
  dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
  dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
  mglat = lat + dlat
  mglng = lng + dlng
  return [lng * 2 - mglng, lat * 2 - mglat]
def mapgps(name):
    data = np.load(f"data/chengdu/npy/{name}.npy", allow_pickle=True)
    res = []
    for d in data:
        if len(d[3])>2:
            after = np.asarray([gcj02_to_wgs84(k[0], k[1]) for k in list(zip(d[3], d[2]))])
            d[2] = after[:, 1]
            d[3] = after[:, 0]
            res.append(d)
    np.save(f'data/chengdu/mapgps/{name}.npy', np.asarray(res, dtype='object'))


    # windows
    d1 = np.load('data/chengdu/origin/20140803.npy', allow_pickle=True)
    with open('chengdu_train.csv', 'w', newline='\n') as f:
        f.write("id;geom\n")
        for indext, item in enumerate(d1):
            f.write(
                f"{indext + 1};LINESTRING({','.join([f'{item[3][i]} {item[2][i]}' for i in range(len(item[3]))])})\n")
def chengduinrec():
    import numpy as np
    import time
    north = 30.75
    south = 30.5930
    east = 104.167
    west = 103.9746
    for name in range(20140804, 20140830):
        try:
            print(f"{name} begin at {time.strftime('%H:%M:%S'.format(time.localtime(time.time())))}")
            data = np.load(f"data/chengdu/mapgps/{name}.npy", allow_pickle=True)
            idx = []
            for i, d in enumerate(data):
                keep = True
                for j in range(len(d[2])):
                    if d[2][j]<south or d[2][j]>north:
                        keep = False
                        break
                    if d[3][j]<west or d[3][j]>east:
                        keep = False
                        break
                if keep:
                    idx.append(i)
            res = data[idx]
            np.save(f'data/chengdu/inrec/{name}.npy', res)
            if name < 20140810:
                with open(f'data/chengdu/inrec/csv/{name}.csv', 'w', newline='\n') as f:
                    f.write("id;geom\n")
                    for indext, item in enumerate(res):
                        f.write(f"{indext + 1};LINESTRING({','.join([f'{item[3][k]} {item[2][k]}' for k in range(len(item[2]))])})\n")
            else:
                with open(f'data/chengdu/inrec/csv2/{name}.csv', 'w', newline='\n') as f:
                    f.write("id;geom\n")
                    for indext, item in enumerate(res):
                        f.write(f"{indext + 1};LINESTRING({','.join([f'{item[3][k]} {item[2][k]}' for k in range(len(item[2]))])})\n")
        except Exception as e:
            print(f"error on {name}")
            print(e)

def test():
    import torch
    from models.TTEModel import TTEModel
    model = TTEModel(87, 52, 52, 30, 3)
    model.load_state_dict(
        torch.load('data/save_models/TTEModel_lstmfinale4norm_porto_TTEModel4/best_model.pkl')['model_state_dict'])
    data = np.load('data/porto/portotrain/test.npy')
    k = data[0]
    res = []
    for i in range(1440):
        k[5] = i
        res.append(model(portoedge([k])[0], {'ba': 2}))

    plt.plot(torch.cat(res).detach().numpy())
    plt.show()
def computedeeptravelmeanstd():
    with open('models/deeptravel/processed_data/train.txt', 'r') as f:
        data = f.readlines()
    import numpy as np
    import json
    data = [json.loads(d) for d in data]
    for k in ['dist_gap', 'time_gap', 'lngs',  'lats',  'time_bin', 'T_X', 'T_Y', 'G_X', 'G_Y']:
        d = np.concatenate([da[k] for da in data])
        print(f'"{k}_mean": {np.mean(d)},')
        print(f'"{k}_std": {np.std(d)},')
    for k in ['dist', 'time']:
        d = [da[k] for da in data]
        print(f'"{k}_mean": {np.mean(d)},')
        print(f'"{k}_std": {np.std(d)},')
    with open('models/deeptravel/DeepTravel/processed_data/val', 'r') as f:
        data = f.readlines()
    for i in range(8):
        with open(f'models/deeptravel/DeepTravel/processed_data/train_{i}', 'w') as f:
            for d in data[int(len(data) * 0.1 * i):int(len(data) * 0.1 * (i + 1))]:
                f.write(d)
    for i in range(8):
        print(f"train_{i},")

def lajideeptrvel():
    import os
    os.chdir('models/deeptravel/DeepTravel')
    from models.deeptravel.DeepTravel.data_loader import *
    import json
    import pickle
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
    loader = get_loader('trainjson.pkl', 128)
    res = []
    for idx, data in enumerate(loader):
        if idx % 100 == 0 and idx > 0:
            with open(f'./processed_data/aftercollate/train_{idx // 100}.pkl', 'wb') as f:
                pickle.dump(res, f)
            print(f"save{idx // 100} at {time.strftime('%H:%M:%S'.format(time.localtime(time.time())))}")
            res.clear()
        res.append(data)