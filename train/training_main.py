import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np

from train.train_model import train_model
from utils.prepare import load_dataset, create_model, create_loss, load_datadict
from utils.metric import calculate_metrics
from utils.util import to_var


def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(enumerate(data_loader))
    for step, (features, truth_data) in tqdm_loader:
        if 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)
        if args.lossinside:
            _, outputs = model(features, truth_data, args, loss_func=None)
        else:
            outputs = model(features, args)
        # outputs = model(features, truth_data=truth_data)

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(len(inds))
        print(inds[0])
        inds = np.concatenate(inds)
    else:
        inds = None
    # print(calculate_metrics(pre2[:, :3], tar2[:, :3], **params))
    # print(calculate_metrics(pre2[:, :6], tar2[:, :6], **params))
    # print(calculate_metrics(pre2[:, :9], tar2[:, :9], **params))
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open('data/result_TTEModel.txt', 'a') as f:
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{metric}\n\n")


def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    # 创建data_loader
    data_loaders, scaler = load_datadict(args)  # todo
    args.scaler = scaler
    model = create_model(args)
    loss_func = create_loss(args)

    model_folder = f"data/save_models/{args.model}_{args.identify}_{args.dataset}"
    tensorboard_folder = f"runs/{args.model}_{args.identify}_{args.dataset}"
    model = model.to(args.device)

    # 训练
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        if os.path.exists(tensorboard_folder):
            shutil.rmtree(tensorboard_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(tensorboard_folder, exist_ok=True)

        if args.optim == "Adam":
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
        elif args.optim == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        elif args.optim == "RMSProp":
            optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
        else:
            raise NotImplementedError()

        train_model(model=model, data_loaders=data_loaders,
                    loss_func=loss_func, optimizer=optimizer,
                    model_folder=model_folder, tensorboard_folder=tensorboard_folder,
                    args=args)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'))['model_state_dict'])
    test_model(model, data_loaders['test'], args)
