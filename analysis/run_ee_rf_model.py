import ee
import utility_functions.model_inputs as uf

ee.Initialize()


def run_ee_rf(method, asset_name):

    dat = uf.get_data()

    if method == 'rz':
        bands = ['tmmn', 'tmmx', 'vpd']
        target_band = 'rzMean'
    elif method == 'surf':
        bands = ['rmax', 'rmin', 'vpd']
        target_band = 'surfMean'
    elif method == 'rz_full':
        bands = ['vpd', 'tmmn', 'tmmx', 'rmin', 'rmax', 'doy', 'latitude', 'longitude']
        target_band = 'rzMean'
    elif method == 'surf_full':
        bands = ['vpd', 'tmmn', 'tmmx', 'rmin', 'rmax', 'doy', 'latitude', 'longitude']
        target_band = 'surfMean'
    else:
        raise ValueError("'method' argument must either be 'rz', 'surf', 'rz_full', or 'surf_full'.")

    training = ee.FeatureCollection('users/colinbrust/train_pts_small')

    classifier = ee.Classifier.randomForest(
        numberOfTrees=50,
        variablesPerSplit=0,
        minLeafPopulation=1,
        outOfBagMode=False
    ).setOutputMode('REGRESSION').train(training, target_band, bands)

    print('Model Trained')

    dates = uf.get_dates(ds='2016-01-01', de='2016-12-31')

    for d in dates:
        while True:
            try:
                # Filter image collection based on the date
                img = dat.filterDate(str(d.date())).first()
                classified = ee.Image(img.classify(classifier).copyProperties(img, ['system:time_start']))

                task = ee.batch.Export.image.toAsset(
                    image=classified,
                    assetId='users/colinbrust/{}/classified_{}'.format(asset_name, str(d.date()).replace('-', '')),
                    description='{} {} Classified'.format(asset_name, str(d.date()).replace('-', '')),
                    region=[[[-126.74328639691237, 50.06495414387699],
                             [-126.74328639691237, 22.757015919458244],
                             [-65.74719264691237, 22.757015919458244],
                             [-65.74719264691237, 50.06495414387699]]],
                    maxPixels=1e13
                )

                task.start()

                print(str(d.date()), 'Classification Submitted')

            except ee.ee_exception.EEException as e:
                print(e)
                continue
            break


run_ee_rf('rz', 'classified_rz')
run_ee_rf('surf', 'classified_surf')
run_ee_rf('rz_full', 'classified_rz_full')
run_ee_rf('surf_full', 'classified_surf_full')
