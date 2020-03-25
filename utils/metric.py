import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        mse = np.nan_to_num(mse)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        mae = np.nan_to_num(mae)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
        mape = np.nan_to_num(mask * mape)
        mape = np.nan_to_num(mape)
        return np.mean(mape)


def calculate_metrics(preds, labels, args = None, null_val=0.0, plot=False, inds=None):
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    try:
        # try:
        #     scaler = args.scaler
        #     preds = scaler.inverse_transform(preds.reshape([-1,1])).squeeze()
        #     # preds = (preds - np.mean(preds))/np.std(preds) * 408.8682+538.480
        #     # preds = (preds - np.mean(preds))/np.std(preds) * 231.2591+490.5749
        #     labels = scaler.inverse_transform(labels.reshape([-1,1])).squeeze()
        # except:
        #     print("no scale")
        # if plot:
        #     plt.scatter(preds, labels)
        #     plt.axis('equal')
        #     plt.show()
        preds = preds.reshape([-1,1]).squeeze()
        labels = labels.reshape([-1,1]).squeeze()
        print(preds[:40000:1905])
        print(labels[:40000:1905])
        # mape = masked_mape_np(preds, labels, 0.0)
        # mae = masked_mae_np(preds, labels, 0.0)
        # rmse = masked_rmse_np(preds, labels, 0.0)
        mape = np.mean(np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5)))
        mse = np.mean(np.square(np.subtract(preds, labels)).astype('float32'))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(np.subtract(preds, labels)).astype('float32'))
        if inds is not None:
            ape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
            res = np.concatenate([inds[ape > mape].reshape(-1,1), ape[ape > mape].reshape(-1,1)], axis=-1)
            np.save('data/porto/badcase.npy', res)
    except Exception as e:
        print(e)
        mae = 0
        mape = 0
        rmse = 0
    try:
        pearsonrs = pearsonr(preds, labels)
    except Exception as e:
        print(e)
        pearsonrs = (0, 0)
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'pearr':pearsonrs[0], 'pearp': pearsonrs[1]}
