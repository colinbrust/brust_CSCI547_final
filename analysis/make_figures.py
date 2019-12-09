import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rio
from sklearn import metrics
import scipy

# Function to calculate r2 correlation
def rsquared(x, y):

    """ Return R^2 where x and y are array-like.
    Taken from https://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy"""
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


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
def map_sm(ras='/Users/cbandjelly/PycharmProjects/sm_ml/plotting_data/sm_ml_fig.tif'):

    # read raster as array
    with rio.open(ras, 'r') as ds:
        arr = ds.read()

    mask = np.where(np.isnan(arr[0]), 0, 1)

    arr *= mask

    arr = np.where(arr == 0, np.nan, arr)

    # different models are stored as bands, extract accordingly
    rz_part, rz_full, surf_part, surf_full, rz, surf = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]

    fig, ax = plt.subplots(2, 3, figsize=(15, 9),
                           sharex=True, sharey=True)

    # First row
    ax[0, 0].imshow(surf, vmin=0, vmax=0.7)
    ax[0, 0].set_title('SMAP L4_SM Surface SM')

    ax[0, 1].imshow(surf_part, vmin=0, vmax=0.7)
    ax[0, 1].set_title('Partial Model Surface SM')

    ax[0, 2].imshow(surf_full, vmin=0, vmax=0.7)
    ax[0, 2].set_title('Full Model Surface SM')

    # Second row
    ax[1, 0].imshow(rz, vmin=0, vmax=0.7)
    ax[1, 0].set_title('SMAP L4_SM Rootzone SM')

    ax[1, 1].imshow(rz_part, vmin=0, vmax=0.7)
    ax[1, 1].set_title('Partial Model Rootzone SM')

    im = ax[1, 2].imshow(rz_full, vmin=0, vmax=0.7)
    ax[1, 2].set_title('Full Model Rootzone SM')

    plt.tight_layout()
    plt.suptitle('Observed and Modeled Soil Moisture (%) on July 1st, 2016')

    fig.subplots_adjust(right=0.6)
    cbar_ax = fig.add_axes([0.1, 0.1, 0.8, 0.01])
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')

    plt.savefig('/Users/cbandjelly/PycharmProjects/sm_ml/figures/SM_map.png', dpi=300)


# Make a table of accuracies to put in paper. Error metrics are calculated for two subregions
# in the midwest and Arizona to see if there are spatial differences in model accuracy.
def make_acc_table(ras1='/Users/cbandjelly/PycharmProjects/sm_ml/plotting_data/az_sm.tif',
                   ras2='/Users/cbandjelly/PycharmProjects/sm_ml/plotting_data/mw_sm.tif'):

    # read in raster as array
    with rio.open(ras1, 'r') as ds:
        arr = ds.read()

    # different models are stored as bands, extract accordingly
    rz_part, rz_full, surf_part, surf_full, rz, surf = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]

    # calculate error metrics on first 3000 pixels
    az_dict = {
     'Arizona_Rootzone_Partial_RMSE': np.sqrt(metrics.mean_squared_error(rz.ravel()[:3000], rz_part.ravel()[:3000])),
     'Arizona_Rootzone_Full_RMSE': np.sqrt(metrics.mean_squared_error(rz.ravel()[:3000], rz_full.ravel()[:3000])),

     'Arizona_Surface_Partial_RMSE': np.sqrt(metrics.mean_squared_error(surf.ravel()[:3000], surf_part.ravel()[:3000])),
     'Arizona_Surface_Full_RMSE': np.sqrt(metrics.mean_squared_error(surf.ravel()[:3000], surf_full.ravel()[:3000])),

     'Arizona_Rootzone_Partial_R2': rsquared(rz.ravel()[:3000], rz_part.ravel()[:3000]),
     'Arizona_Rootzone_Full_R2': rsquared(rz.ravel()[:3000], rz_full.ravel()[:3000]),

     'Arizona_Surface_Partial_R2': rsquared(surf.ravel()[:3000], surf_part.ravel()[:3000]),
     'Arizona_Surface_Full_R2': rsquared(surf.ravel()[:3000], surf_full.ravel()[:3000])}

    # read in raster as array
    with rio.open(ras2, 'r') as ds:
        arr = ds.read()

    # different models are stored as bands, extract accordingly
    rz_part, rz_full, surf_part, surf_full, rz, surf = arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]


    # calculate error metrics on first 3000 pixels
    mw_dict={
    'Midwest_Rootzone_Partial_RMSE': np.sqrt(metrics.mean_squared_error(rz.ravel()[:3000], rz_part.ravel()[:3000])),
    'Midwest_Rootzone_Full_RMSE': np.sqrt(metrics.mean_squared_error(rz.ravel()[:3000], rz_full.ravel()[:3000])),

    'Midwest_Surface_Partial_RMSE': np.sqrt(metrics.mean_squared_error(surf.ravel()[:3000], surf_part.ravel()[:3000])),
    'Midwest_Surface_Full_RMSE': np.sqrt(metrics.mean_squared_error(surf.ravel()[:3000], surf_full.ravel()[:3000])),

    'Midwest_Rootzone_Partial_R2': rsquared(rz.ravel()[:3000], rz_part.ravel()[:3000]),
    'Midwest_Rootzone_Full_R2': rsquared(rz.ravel()[:3000], rz_full.ravel()[:3000]),

    'Midwest_Surface_Partial_R2': rsquared(surf.ravel()[:3000], surf_part.ravel()[:3000]),
    'Midwest_Surface_Full_R2': rsquared(surf.ravel()[:3000], surf_full.ravel()[:3000])}

    az_df = pd.Series(az_dict).to_frame().reset_index()
    mw_df = pd.Series(mw_dict).to_frame().reset_index()

    df = pd.concat([az_df, mw_df])
    df.columns = ['id', 'Value']

    df[['Location', 'Soil Location', 'Model', 'Error Metric']] = df['id'].str.split('_', expand=True)
    df = df.drop(columns=['id'])
    df = df[['Location', 'Soil Location', 'Model', 'Error Metric', 'Value']]

    return df.to_latex()


