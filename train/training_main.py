import os
import shutil

import torch
from torch import optim
import numpy as np
from tqdm import tqdm

from train.train_model import train_model
from utils.load_config import get_attribute
from utils.loss import MSELoss, BCELoss
from utils.data_container import load_dataset
from utils.util import load_graph_data, convert_to_gpu
from utils.loss import masked_rmse_loss, masked_mse_loss
from utils.metric import calculate_metrics

from utils.util import cheb_poly_approx, calculate_scaled_laplacian, create_kernel, calculate_random_walk_matrix


from models.dcrnn_model import DCRNNModel
from models.notuse.stgcn import STGCN
from models.transformerg import MyModel
from models.notuse.transformer_pytorch import Transformer_pytorch
from models.wavenet import gwnet


def create_single_kernel(adj, type):
    if type == 'random':
        return calculate_random_walk_matrix(adj).T
    elif type == 'cheb':
        return calculate_scaled_laplacian(adj, lambda_max=None)
    else:
        return calculate_scaled_laplacian(adj, lambda_max=None)


def create_model(model_name, **kwargs):
    if model_name == 'dcrnn':
        enc_input_dim = get_attribute((model_name, 'enc_input_dim'))
        dec_input_dim = get_attribute((model_name, 'dec_input_dim'))
        kernel_size = get_attribute((model_name, 'kernel_size'))
        graph_pkl_filename = get_attribute((model_name, 'adj_path'))
        _, _, adj = load_graph_data(graph_pkl_filename)
        graph_conv_type = get_attribute((model_name, 'graph_conv_type'))
        adj = create_single_kernel(adj, graph_conv_type)
        support = create_kernel(adj, kernel_size)
        num_nodes = get_attribute((model_name, 'num_nodes'))
        num_rnn_layers = get_attribute((model_name, 'num_rnn_layers'))
        rnn_units = get_attribute((model_name, 'rnn_units'))
        seq_input = get_attribute((model_name, 'seq_input'))
        seq_output = get_attribute((model_name, 'seq_output'))
        output_dim = get_attribute((model_name, 'output_dim'))
        cl_decay = get_attribute((model_name, 'cl_decay'))
        return DCRNNModel(support, enc_input_dim, dec_input_dim, num_nodes, num_rnn_layers,
                          rnn_units, seq_input, seq_output, output_dim), {'cl_decay': cl_decay}
    elif model_name == 'stgcn':
        adj_ks = get_attribute((model_name, 'ks'))
        _, _, adj = load_graph_data(get_attribute((model_name, 'adj_path')))

        support = convert_to_gpu(torch.FloatTensor(cheb_poly_approx(adj, adj_ks, adj.shape[0])))

        temporal_kernel = get_attribute((model_name, 'kt'))
        num_timesteps_input = get_attribute((model_name, 'num_timesteps_input'))
        num_timesteps_output = get_attribute((model_name, 'num_timesteps_output'))
        num_features = get_attribute((model_name, 'num_features'))
        return STGCN(support.shape[0], num_features, num_timesteps_input, num_timesteps_output,
                     temporal_kernel, adj_ks, support), {}
    elif model_name == 'transformer':
        blocks = get_attribute((model_name, 'blocks'))
        seq_input = get_attribute((model_name, 'seq_input'))
        seq_output = get_attribute((model_name, 'seq_output'))
        num_nodes = get_attribute((model_name, 'num_nodes'))
        features = get_attribute((model_name, 'features'))
        hiddendim = get_attribute((model_name, 'hiddendim'))

        kernel_size = get_attribute((model_name, 'kernel_size'))
        _, _, adj = load_graph_data(get_attribute((model_name, 'adj_path')))

        graph_conv_type = get_attribute((model_name, 'graph_conv_type'))
        adj = create_single_kernel(adj, graph_conv_type)
        support = create_kernel(adj, kernel_size)
        print(support.shape)
        if graph_conv_type == 'None':
            support = None
        return MyModel(blocks, seq_input, seq_output, num_nodes, features, hiddendim, 0, support=support, adaptive=True), {}
    elif model_name == "gwave":
        device = get_attribute("cuda")
        return gwnet(device, 207), {}


    ########################
    elif model_name == 'transformer_pytorch':
        return Transformer_pytorch(input_size=228, nheads=4, num_encoders=2, num_decoders=2, d_inner=512), {}
    ########################


def create_loss(loss_type, **kwargs):
    if loss_type == 'mse_loss':
        return convert_to_gpu(MSELoss())
    elif loss_type == 'bce_loss':
        return convert_to_gpu(BCELoss())
    elif loss_type == 'masked_rmse_loss':
        return masked_rmse_loss(kwargs['scaler'], 0.0)
    elif loss_type == 'masked_mse_loss':
        return masked_mse_loss(kwargs['scaler'], 0.0)
    else:
        raise ValueError("Unknown loss function.")


def test_model(model, data_loader, **params):
    # todo single step
    model.eval()
    predictions = list()
    targets = list()
    params['is_eval'] = True
    tqdm_loader = tqdm(enumerate(data_loader))
    for step, (features, truth_data) in tqdm_loader:
        features = convert_to_gpu(features)
        truth_data = convert_to_gpu(truth_data)
        outputs = model(features, **params)
        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    print(calculate_metrics(pre2[:, :3], tar2[:, :3], **params))
    print(calculate_metrics(pre2[:, :6], tar2[:, :6], **params))
    print(calculate_metrics(pre2[:, :9], tar2[:, :9], **params))
    print(calculate_metrics(pre2, tar2, **params))


def train_main():
    batchsize = get_attribute('batch_size')  # todo argparser
    modelname = get_attribute('model_name')
    save_name = get_attribute('save_name')
    print(f"train {modelname}+{save_name}")
    # 创建data_loader
    data_dirs = get_attribute((modelname, 'data_dirs'))
    data_loaders, scaler, modelparams = load_dataset(modelname, data_dirs, batchsize, get_attribute((modelname, "single"))==1)
    model, params = create_model(modelname, **modelparams)
    params['scaler'] = scaler

    model_folder = f"data/save_models/{modelname}+{save_name}"
    tensorboard_folder = f"runs/{modelname}+{save_name}"
    model = convert_to_gpu(model)

    # 训练
    if get_attribute('mode') == 'train':
        loss_func = create_loss(loss_type=get_attribute('loss_function'), **{'scaler': scaler})
        num_epochs = get_attribute('epoch')
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        if os.path.exists(tensorboard_folder):
            shutil.rmtree(tensorboard_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)
        os.makedirs(tensorboard_folder, exist_ok=True)

        if get_attribute("optim") == "Adam":
            optimizer = optim.Adam(model.parameters(),
                                   lr=get_attribute("learning_rate"),
                                   weight_decay=get_attribute("weight_decay"))
        elif get_attribute("optim") == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=get_attribute("learning_rate"),
                                  momentum=0.9)
        elif get_attribute("optim") == "RMSProp":
            optimizer = optim.RMSprop(model.parameters(),
                                      lr=get_attribute("learning_rate"))
        else:
            raise NotImplementedError()

        train_model(model=model,
                    data_loaders=data_loaders,
                    loss_func=loss_func,
                    optimizer=optimizer,
                    num_epochs=num_epochs,
                    model_folder=model_folder,
                    tensorboard_folder=tensorboard_folder,
                    **params)
    model.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'))['model_state_dict'])
    test_model(model, data_loaders['test'], **params)
