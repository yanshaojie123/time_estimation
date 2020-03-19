import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
# 衡量模型的性能
# def evaluate(y_predictions: np.ndarray, y_targets: np.ndarray, threshold: float = 0.5):
#     """
#     :param y_predictions: a 1d array, with length as number of samples
#     :param y_targets: a 1d array, with length as number of samples
#     :param threshold: threshold, default 0.5
#     :return:
#     """
#     assert y_predictions.shape == y_targets.shape, \
#         f'Predictions of shape {y_predictions.shape} while targets of shape {y_predictions.shape}.'
#     mse = mean_squared_error(y_targets, y_predictions)
#     rmse = mse ** 0.5
#     mae = mean_absolute_error(y_targets, y_predictions)
#     # pcc, p_value = pearsonr(y_predictions, y_targets)
#
#     y_predictions = y_predictions >= threshold
#     y_targets = y_targets == 1
#
#     correct = (y_predictions == y_targets).sum()
#     # accuracy = correct / len(y_predictions)
#
#     tp = ((y_predictions == 1) & (y_targets == 1)).sum()
#     fp = ((y_predictions == 1) & (y_targets == 0)).sum()
#     fn = ((y_predictions == 0) & (y_targets == 1)).sum()
#
#     # precision = tp / (tp + fp)
#     # recall = tp / (tp + fn)
#     # f1_score = 2 * (precision * recall) / (precision + recall)
#
#     del y_predictions, y_targets, correct, tp, fp, fn, threshold
#
#     return {key.upper().replace('_', '-'): val for key, val in locals().items()}


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


# def masked_mae_loss(null_val):
#     def loss(preds, labels):
#         mae = masked_mae_torch(preds=preds, labels=labels, null_val=null_val)
#         return mae
#     return loss


def calculate_metrics(preds, labels, args = None, null_val=0.0, plot=False, inds=None):  # todo: delete one from this and evaluate()
    """
    Calculate the MAE, MAPE, RMSE
    :param df_pred:
    :param df_test:
    :param null_val:
    :return:
    """
    try:
        # print(pearsonr(preds, labels))
        # print(1)
        try:
            scaler = args.scaler
            preds = scaler.inverse_transform(preds.reshape([-1,1])).squeeze()
            # print(2)
            # preds = (preds - np.mean(preds))/np.std(preds) * 231.2591+490.5749
            # preds = (preds - np.mean(preds))/np.std(preds) * 227.6324 + 495.88559451758
            # print(3)
            labels = scaler.inverse_transform(labels.reshape([-1,1])).squeeze()
        except:
            print("no scale")
        # print(4)
        # if plot:
        #     plt.scatter(preds, labels)
        #     plt.axis('equal')
        #     plt.show()
        print(preds[:10])
        print(labels[:10])
        mape = masked_mape_np(preds, labels, 0.0)
        mae = masked_mae_np(preds, labels, 0.0)
        rmse = masked_rmse_np(preds, labels, 0.0)
        if inds is not None:
            ape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels + 1e-5))
            res = np.concatenate([inds[ape > mape].reshape(-1,1), ape[ape > mape].reshape(-1,1)], axis=-1)
            np.save('data/porto/badcase.npy', res)

        # mape = np.mean(np.abs((preds-labels)/(labels+1e-5)))
        # mae = np.mean(np.abs(preds-labels).astype('float32'))
        # rmse = np.sqrt(np.mean((preds-labels)**2))
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
    # return mae, mape, rmse
    # print(pearsonrs)
    return {'MAE': mae, 'MAPE': mape, 'RMSE': rmse, 'pearr':pearsonrs[0], 'pearp': pearsonrs[1]}
