import json
import torch
import logger

from models.DeepTravel import DeepTravel

import utils
import data_loader


def train(model, train_set, eval_set, dt_logger):

    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # for epoch in range(num_of_epochs):

    for data_file in train_set:

        model.train()

        data_iter = data_loader.get_loader(data_file, 1)

        running_loss = 0.0

        for idx, (stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers) in enumerate(data_iter):

            stats, temporal, spatial, dr_state = utils.to_var(stats), utils.to_var(temporal), utils.to_var(spatial), utils.to_var(dr_state)
            short_ttf, long_ttf = utils.to_var(short_ttf), utils.to_var(long_ttf)

            loss = model.evaluate(stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers)
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.mean().data.item()


def main():
    config = json.load(open('./config.json', 'r'))
    dt_logger = logger.get_logger()

    model = DeepTravel()

    train(model, config['train_set'], config['eval_set'], dt_logger)


if __name__ == '__main__':
    main()
