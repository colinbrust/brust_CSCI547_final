import ee
import datetime as dt
import pandas as pd
import numpy as np
import utility_functions.model_inputs as uf

ee.Initialize()


def extract_random_pts():

    roi = uf.get_roi()

    stack = uf.get_data()

    date_start = dt.datetime.strptime('2016-01-01', '%Y-%m-%d')
    date_end = dt.datetime.strptime('2018-12-31', '%Y-%m-%d')
    diff = date_end - date_start
    dates = [date_start + dt.timedelta(days=x) for x in range(diff.days + 1)]

    df_out = pd.DataFrame()

    i = 1

    for d in dates:

        while True:
            try:
                rand_seed = np.random.randint(low=0, high=20000)
                pts = ee.FeatureCollection.randomPoints(
                    region=roi,
                    points=500,
                    seed=rand_seed
                )

                img = stack.filterDate(str(d.date())).first()

                plot_sample_regions = img.sampleRegions(
                    collection=pts,
                    scale=9000
                )

                data = plot_sample_regions.getInfo()

                df = pd.DataFrame([x['properties'] for x in data['features']])
                df_out = pd.concat([df_out, df], axis=0, ignore_index=True)

                if i % 50 == 0:
                    df_out.to_csv('/mnt/e/Data/sm_ml_data/temp_dat_{}.csv'.format(str(i)), index=False)

                print(d, i)
                i += 1
            except ee.ee_exception.EEException as e:
                print(e)
                continue
            break

    df_out.to_csv('/mnt/e/Data/sm_ml_data/smap_gm_data.csv')


extract_random_pts()
