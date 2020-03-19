import json
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from models.deeptravel.DeepTravel import logger

from models.deeptravel.DeepTravel.models.DeepTravel import DeepTravel

from models.deeptravel.DeepTravel import utils
from models.deeptravel.DeepTravel import data_loader
import copy
from tqdm import tqdm
import time
from utils.metric import calculate_metrics
import numpy as np


def train(model, train_set, eval_set, dt_logger):

    if torch.cuda.is_available():
        model.cuda()
    print("train_begin "+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # train_iter = data_loader.get_loader(train_set[0], 128)
    # val_iter = data_loader.get_loader(eval_set[0], 128)
    num_of_epochs = 10
    for epoch in range(num_of_epochs):
        model.train()
        running_loss = 0.0
        for data_file in train_set:
            train_iter = data_loader.get_loader(data_file, 1)
        # if True:
            tqdm_loader = tqdm(enumerate(train_iter))
            datatime = 0
            modeltime = 0
            backtime = 0
            predictions, targets = list(), list()
            for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in tqdm_loader:
                try:
                # if True:
                    start = time.time()
                    stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(spatial), utils.to_var(dr_state)
                    short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)

                    dt = time.time()
                    datatime += dt-start

                    loss, pred, target = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
                    # print(loss.shape)
                    predictions.append(pred)
                    targets.append(target)
                    mt = time.time()
                    modeltime += mt-dt

                    optimizer.zero_grad()
                    loss.sum().backward()
                    optimizer.step()
                    running_loss += loss.mean().data.item()
                    bt = time.time()
                    backtime += bt-mt
                    tqdm_loader.set_description(
                        f'epoch: {epoch}, loss: {running_loss / (idx + 1)}')
                except Exception as e:
                    print(e)
                # if idx % 100 ==0:
                # print(f"datatime:{datatime/(idx+1)}")
                # print(f"modeltime:{modeltime/(idx+1)}")
                # print(f"backtime:{backtime/(idx+1)}")
            score = calculate_metrics(np.asarray(predictions), np.asarray(targets))
            print(score)
        torch.save(copy.deepcopy(model.state_dict()), f'./models/deeptravel/DeepTravel/save/{epoch}.pkl')

    # for data_file in train_set:
    #
    #     model.train()
    #
    #     data_iter = data_loader.get_loader(data_file, 1)
    #
    #     running_loss = 0.0
    #
    #     for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in enumerate(data_iter):
    #
    #         stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(spatial), utils.to_var(dr_state)
    #         short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)
    #
    #         loss = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
    #         optimizer.zero_grad()
    #         loss.sum().backward()
    #         optimizer.step()
    #
    #         running_loss += loss.mean().data.item()


def main():
    config = json.load(open('./models/deeptravel/DeepTravel/config.json', 'r'))
    dt_logger = logger.get_logger()

    model = DeepTravel()

    train(model, config['train_set'], config['eval_set'], dt_logger)


if __name__ == '__main__':
    main()
