import pandas as pd
import numpy as np
import pickle

def edgeinfo():
    edgeinfo = pd.read_csv('data/porto/fmm/original_edges.csv')
    edgemap = pd.read_csv('data/porto/fmm/edgemap.csv')  # todo oid
    edgemap = edgemap.drop(['geom', 'source', 'target'], axis=1)
    result = pd.merge(edgemap, edgeinfo, left_on='oid', right_on='gid').sort_values(by='gid_x')
    result.to_csv('data/porto/edgeinfo.csv')

# def edgeinfo():
    edgeinfo = pd.read_csv('data/porto/fmm/edgeinfo.csv')
    edgedata = edgeinfo.values
    result = {}
    # highway length lanes maxspeed width bridge tunnel
    for k in edgedata:
        result[k[1]] = (k[6], k[8], k[12], k[14], k[16], k[17], k[18])
    with open('data/porto/edgeinfo.pkl', 'wb') as f:
        import pickle
        pickle.dump(result, f)

def badcase():
    data = np.load('data/porto/portotrain/test.npy')
    mapd = {data[i][0]: i for i in range(47785)}
    bad = np.load('data/porto/badcase.npy').tolist()
    print(data[mapd[1275177]])

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

def edgegps():
    import pickle
    import pandas as pd
    import os
    os.chdir("data/porto")
    with open("chengdu_edgeinfo.pkl", 'rb') as f:
        edgeinfo = pickle.load(f)
    gpsdata = pd.read_csv("chengdu_dual_withgps.csv")
    va = gpsdata.values
    for v in va:
        v[2] = v[2].replace('LINESTRING(', "")
        v[2] = v[2].replace(")", "")
    gpss = {}
    for v in va:
        gpss[v[0]] = [float(d) for d in v[2].split(',')[0].split(' ')] + [float(d) for d in v[2].split(',')[-1].split(' ')]
    for k in edgeinfo.keys():
        edgeinfo[k] = list(edgeinfo[k]) + gpss[k]
    with open("chengdu_edgeinfo.pkl", 'wb') as f:
        pickle.dump(edgeinfo, f)

def norm():
    from sklearn.preprocessing import StandardScaler
    import pickle
    with open('../edgeinfodict.pkl', "rb") as f:
        edgeinfo = pickle.load(f)

    data = np.load("train.npy", allow_pickle=True)
    scale = StandardScaler()
    lala = []
    for d in data:
        red = 0
        for dd in d[2]:
            temp = []
            info = edgeinfo[dd]
            temp.append(info[1])
            red += info[1]
            temp.append(red)
            temp += info[7:11]
            lala.append(temp)
    scale.fit(lala)
    print(scale.mean_)
    print(scale.scale_)
    scale.transform()