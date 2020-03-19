import pandas as pd
import numpy as np
import math
from math import pi
import time

def chengdutopath(day):
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

# fmm

def chengdu_dup_hlratio():
    import pandas as pd
    import numpy as np
    import os
    os.chdir('~/Workspace/fmm-master/chengdu/txt')
    def duppath(x):
        if type(x) == type(" "):
            try:
                k = {}
                links = x.split(',')
                for i in range(len(links)-1, -1, -1):
                    link = links[i]
                    if link not in k.keys():
                        k[link] = i
                    else:
                        links = links[:i] + links[k[link]:]
                        return duppath(','.join(links))
            except:
                return np.nan
        return x
    for date in ["20140808"]:
        print(date)
        data = pd.read_csv(f"{date}.txt", sep=';')
        data.cpath = data.cpath.map(duppath)
        # data.mgeom = data.mgeom.map(depmgeom)
        # data.to_csv('data/porto/train_drop.csv', sep=';')

        data = data.values
        data = data[list(map(lambda x: type(x) == type(" "), data[:, 2]))]  # drop nan
        out = 0
        ind = []
        # drop losing path
        for i, d in enumerate(data):
            o = set(d[1].split(','))
            c = d[2].split(',')
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
        l2 = train_res[:, 1]
        l2 = [[int(k) for k in l.split(',')] for l in l2]
        train_res[:, 1] = l2
        l2 = train_res[:, 2]
        l2 = [[int(k) for k in l.split(',')] for l in l2]
        train_res[:, 2] = l2
        data = train_res[:, :3]

        ratio = np.asarray([len(d[2]) / len(d[1]) for d in data])
        ratioforind = np.asarray([len(d[2]) / len(d[1]) for d in data])
        ratio.sort()
        low = ratio[int(ratio.shape[0] * 0.05)]
        high = ratio[int(ratio.shape[0] * 0.95)]
        ind = np.logical_and(ratioforind > low, ratioforind < high)
        data = data[ind]

        train_data = data[:int(len(data) * 0.8)]
        np.save(f'afterprocess/{date}_train.npy', train_data)
        val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
        # val_data = data[data.id >= len(data) * 0.8][data.id < len(data)*0.9].drop('length', axis=1).values
        np.save(f'afterprocess/{date}_val.npy', val_data)
        test_data = data[int(len(data) * 0.9):]
        np.save(f'afterprocess/{date}_test.npy', test_data)

def chengdu_time_deeptte():
    import pandas as pd
    import numpy as np
    import os
    import time
    for date in ["20140803", "20140804", "20140805", "20140806", "20140808", "20140809"]:
        print(date)
        traindata = np.load(f"/home/hanlz/Workspace/fmm-master/chengdu/txt/afterprocess/{date}_train.npy",
                            allow_pickle=True)
        valdata = np.load(f"/home/hanlz/Workspace/fmm-master/chengdu/txt/afterprocess/{date}_val.npy",
                            allow_pickle=True)
        testdata = np.load(f"/home/hanlz/Workspace/fmm-master/chengdu/txt/afterprocess/{date}_test.npy",
                          allow_pickle=True)
        dataori = np.load(f"/home/hanlz/Workspace/time_estimation/data/chengdu/inrec/{date}.npy", allow_pickle=True)

        for data, name in [(traindata, "train"), (valdata, "val"), (testdata, "test")]:
            index = data[:, 0]
            deeptte = dataori[list(index-1)]
            np.save(f"/home/hanlz/Workspace/time_estimation/data/chengdu/deeptte/{date}_{name}.npy", deeptte)
            wday = list(map(lambda x: time.localtime(x).tm_wday, deeptte[:, 1]))
            yday = list(map(lambda x: time.localtime(x).tm_yday, deeptte[:, 1]))
            timet = list(map(lambda x: time.localtime(x).tm_hour*60+time.localtime(x).tm_min, deeptte[:, 1]))
            timef = list(map(lambda x: x[-1], deeptte[:, 4]))
            res = np.concatenate([data, np.asarray(wday).reshape(-1,1), np.asarray(yday).reshape(-1,1), np.asarray(timet).reshape(-1,1), np.asarray(timef).reshape(-1,1)], axis = -1)
            np.save(f"/home/hanlz/Workspace/time_estimation/data/chengdu/TTEModel/{date}_{name}.npy", res)

def unite():
    import os
    os.chdir('/home/hanlz/Workspace/time_estimation/data/chengdu/TTEModel')
    traindatas = []
    valdatas = []
    testdatas = []
    for date in [20140803, 20140804, 20140805, 20140806, 20140808]:
        traindatas.append(np.load(f'{date}_train.npy', allow_pickle=True))
        valdatas.append(np.load(f'{date}_val.npy', allow_pickle=True))
        testdatas.append(np.load(f'{date}_test.npy', allow_pickle=True))
    trainres = np.concatenate(traindatas)
    np.save("train.npy", trainres)
    valres = np.concatenate(valdatas)
    np.save("val.npy", valres)
    testres = np.concatenate(testdatas)
    np.save("test.npy", testres)


