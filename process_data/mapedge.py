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



