import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
import sklearn as skl

ee.Initialize()


# plot timeseries of predicted rz and surf sm relative to SMAP timeseries
def plot_timeseries(df='/mnt/e/PycharmProjects/sm_ml/plotting_data/l4_ts.csv'):

    df = pd.read_csv(df)

    fig, axs = plt.subplots(2, sharex=True, sharey=True)
    axs[0].plot(df.L4_rz.values, label='L4 Rootzone SM')
    axs[0].plot(df.L4_surf.values, label='L4 Surface SM')
    axs[0].set_title('SMAP L4_SM SM Timeseries')
    axs[1].plot(df.pred_rz.values, label='Pred. Rootzone SM')
    axs[1].plot(df.pred_surf.values, label='Pred. Surface SM')
    axs[1].set_title('RF SM Prediction Timeseries')

    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    axs[1].set_xlabel('Days Since Jan 1, 2016')
    plt.savefig('/mnt/e/PycharmProjects/sm_ml/figures/SM_TS.png', 300)


# map the different methods of SM estimates for the CONUS domain.
def map_sm(ras=''):

    with rio.open(ras, 'r') as ds:
        arr = ds.read()

    return None


# make a table of accuracies to put in paper
def make_acc_table(ras1='/mnt/e/PycharmProjects/sm_ml/plotting_data/az_sm.tif',
                   ras2='/mnt/e/PycharmProjects/sm_ml/plotting_data/mw_sm.tif'):

    with rio.open(ras1, 'r') as ds:
        arr = ds.read()

    rz_part, rz_full, surf_part, surf_full, rz, surf = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]

    az_rzp_rmse = skl.metrics.mean_squared_error(rz, rz_part)
    az_rzf_rmse = skl.metrics.mean_squared_error(rz, rz_full)

    az_

    with rio.open(ras1, 'r') as ds:
        arr = ds.read()

    return None


