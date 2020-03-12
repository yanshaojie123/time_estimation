import json
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import logger

from models.DeepTravel import DeepTravel

import utils
import data_loader
import copy
from tqdm import tqdm
import time

def train(model, train_set, eval_set, dt_logger):

    if torch.cuda.is_available():
        model.cuda()
    print("train_begin "+time.strftime("%H:%M:%S".format(time.localtime(time.time()))))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
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
            for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in tqdm_loader:
                try:
                # if True:
                    start = time.time()
                    stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(spatial), utils.to_var(dr_state)
                    short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)

                    dt = time.time()
                    datatime += dt-start

                    loss = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
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
            print(f"epoch:{epoch} loss:{running_loss}")
        torch.save(copy.deepcopy(model.state_dict()), f'./save/{epoch}.pkl')

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
    config = json.load(open('./config.json', 'r'))
    dt_logger = logger.get_logger()

    model = DeepTravel()

    train(model, config['train_set'], config['eval_set'], dt_logger)


if __name__ == '__main__':
    main()
