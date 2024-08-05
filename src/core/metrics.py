import numpy as np


def compute_metrics(pose_err_all):
    mae  = np.average(np.abs(pose_err_all), axis=0)
    mse  = np.average(np.square(pose_err_all), axis=0)
    rmse = np.sqrt(mse)
    return mae, mse, rmse