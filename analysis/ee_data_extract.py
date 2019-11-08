import ee
import datetime as dt
import pandas as pd
import numpy as np
import utility_functions.model_inputs as uf

ee.Initialize()


def extract_random_pts():

     # import CONUS roi
    roi = uf.get_roi()

    # create imageCollection of Gridmet meteorology and SMAP soil moisture data
    stack = uf.get_data()

    # Create list of dates to iterate over
    date_start = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    date_end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
    diff = date_end - date_start
    dates = [date_start + dt.timedelta(days=x) for x in range(diff.days + 1)]

    df_out = pd.DataFrame()

    i = 1

    for d in dates:

        # EarthEngine tends to throw random memory errors.
        # Use this loop to prevent a memory error from stopping analysis.
        while True:
            try:
                # EarthEngine uses a default seed for the randomPoints function, so generate a new random number
                # at every iteration so that every day has a different sample of points.
                rand_seed = np.random.randint(low=0, high=20000)
                # Create 500 random points across CONUS
                pts = ee.FeatureCollection.randomPoints(
                    region=roi,
                    points=500,
                    seed=rand_seed
                )

                # Filter image collection based on the date
                img = stack.filterDate(str(d.date())).first()

                # Sample the pixels underneath random points
                plot_sample_regions = img.sampleRegions(
                    collection=pts,
                    scale=9000
                )

                # Extract the data to a Pandas DataFrame
                data = plot_sample_regions.getInfo()

                df = pd.DataFrame([x['properties'] for x in data['features']])
                df_out = pd.concat([df_out, df], axis=0, ignore_index=True)

                print(d, i)
            except ee.ee_exception.EEException as e:
                print(e)
                continue
            break

    df_out.to_csv('/mnt/e/Data/sm_ml_data/smap_gm_data.csv')


extract_random_pts()
