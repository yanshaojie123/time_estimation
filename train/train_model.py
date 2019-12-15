import copy
import time
from typing import Dict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, convert_to_gpu


def train_model(model: nn.Module,
                data_loaders: Dict[str, DataLoader],
                loss_func: callable,
                optimizer,
                num_epochs,
                model_folder: str,
                tensorboard_folder: str,
                **kwargs):
    phases = ['train', 'val', 'test']

    writer = SummaryWriter(tensorboard_folder)

    since = time.clock()

    # loss_func = convert_to_gpu(loss_func)  # Todo

    save_dict, worst_rmse = {'model_state_dict': copy.deepcopy(model.state_dict()), 'epoch': 0}, 100000

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.2, patience=5, threshold=1e-3, min_lr=1e-6)

    try:
        for epoch in range(num_epochs):
            running_loss = {phase: 0.0 for phase in phases}
            for phase in phases:
                if phase == 'train':
                    kwargs['is_eval'] = False
                    model.train()
                else:
                    kwargs['is_eval'] = True
                    model.eval()

                steps, predictions, targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(data_loaders[phase]))
                for step, (features, truth_data) in tqdm_loader:
                    # if step == 0 and epoch == 0:
                    #     print(features.shape)  64, 60, 207, 2
                    #     print(truth_data.shape)  64, 12, 207
                    # print(truth_data)
                    # print(features.shape)
                    features = convert_to_gpu(features)
                    truth_data = convert_to_gpu(truth_data)
                    global_step = (epoch) * 2500 + steps/10
                    kwargs['global_step'] = global_step
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(features, **kwargs)
                        # outputs = torch.squeeze(outputs)  # squeeze [batch-size, 1] to [batch-size]
                        # print(outputs)
                        loss = loss_func(truth=truth_data, predict=outputs)
                        # loss = loss_func(outputs, truth_data)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    targets.append(truth_data.cpu().numpy())
                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())

                    running_loss[phase] += loss * truth_data.size(0)
                    steps += truth_data.size(0)

                    tqdm_loader.set_description(
                        f'{phase} epoch: {epoch}, {phase} loss: {running_loss[phase] / steps}')

                    # For the issue that the CPU memory increases while training. DO NOT know why, but it works.
                    torch.cuda.empty_cache()

                # 性能
                predictions = np.concatenate(predictions)
                targets = np.concatenate(targets)
                # print(predictions[:3, :3])
                # print(targets[:3, :3])
                scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1), targets.reshape(targets.shape[0], -1), **kwargs)
                writer.add_scalars(f'score/{phase}', scores, global_step=epoch)
                print(scores)
                if phase == 'val' and scores['RMSE'] < worst_rmse:
                    worst_rmse = scores['RMSE'],
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                     epoch=epoch,
                                     optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))

            scheduler.step(running_loss['train'])

            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(data_loaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    finally:
        time_elapsed = time.clock() - since
        print(f"cost {time_elapsed} seconds")

        save_model(f"{model_folder}/best_model.pkl", **save_dict)
        save_model(f"{model_folder}/final_model.pkl",
                   **{'model_state_dict': copy.deepcopy(model.state_dict()),
                      'epoch': num_epochs,
                      'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())})
