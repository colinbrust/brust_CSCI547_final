import ee

ee.Initialize()

# export the map that will be used in the map_sm function in 'make_figures'
def export_map_data():
    a = ee.ImageCollection("users/colinbrust/classified_rz")
    b = ee.ImageCollection("users/colinbrust/classified_rz_full")
    c = ee.ImageCollection("users/colinbrust/classified_surf")
    d = ee.ImageCollection("users/colinbrust/classified_surf_full")
    x = ee.ImageCollection("users/colinbrust/l4rz_2016")

    roi = ee.Geometry.Polygon(
        [[[-126.74328639691237, 50.06495414387699],
          [-126.74328639691237, 22.757015919458244],
          [-65.74719264691237, 22.757015919458244],
          [-65.74719264691237, 50.06495414387699]]], None, False)

    x = x.select(['rzMean', 'surfMean']).filterDate('2016-07-01', '2016-07-02').first()
    a = a.filterDate('2016-07-01', '2016-07-02').first().rename('rz_part').updateMask(x.select('rzMean'))
    b = b.filterDate('2016-07-01', '2016-07-02').first().rename('rz_full').updateMask(x.select('rzMean'))
    c = c.filterDate('2016-07-01', '2016-07-02').first().rename('surf_part').updateMask(x.select('rzMean'))
    d = d.filterDate('2016-07-01', '2016-07-02').first().rename('surf_full').updateMask(x.select('rzMean'))

    out = a.addBands(b).addBands(c).addBands(d).addBands(x)

    task = ee.batch.Export.image.toDrive(
        image=out,
        description='sm_ml_fig',
        folder='GEE_Exports',
        region=roi,
        scale=9000,
        crs='EPSG:4326'
    )
    task.start()


# export annual means for arizona and midwest subregions to use as input into error tables  in 'make_figures'.
def export_mean_annual():
    az_roi = ee.Geometry.Polygon(
        [[[-114.17586103472672, 36.30452158073394],
          [-114.17586103472672, 32.313155408546024],
          [-109.07820478472672, 32.313155408546024],
          [-109.07820478472672, 36.30452158073394]]], None, False)

    midwest_roi = ee.Geometry.Polygon(
        [[[-89.56648603472672, 39.595549379376116],
          [-89.56648603472672, 35.019220702712595],
          [-82.49129072222672, 35.019220702712595],
          [-82.49129072222672, 39.595549379376116]]], None, False)

    a = ee.ImageCollection("users/colinbrust/classified_rz")
    b = ee.ImageCollection("users/colinbrust/classified_rz_full")
    c = ee.ImageCollection("users/colinbrust/classified_surf")
    d = ee.ImageCollection("users/colinbrust/classified_surf_full")
    x = ee.ImageCollection("users/colinbrust/l4rz_2016")

    x = x.select(['rzMean', 'surfMean']).mean()
    a = a.mean().rename('rz_part').updateMask(x.select('rzMean'))
    b = b.mean().rename('rz_full').updateMask(x.select('rzMean'))
    c = c.mean().rename('surf_part').updateMask(x.select('rzMean'))
    d = d.mean().rename('surf_full').updateMask(x.select('rzMean'))

    out = a.addBands(b).addBands(c).addBands(d).addBands(x)

    task = ee.batch.Export.image.toDrive(
        image=out.clip(az_roi),
        description='az_sm',
        folder='GEE_Exports',
        region=[[[-114.17586103472672, 36.30452158073394],
                 [-114.17586103472672, 32.313155408546024],
                 [-109.07820478472672, 32.313155408546024],
                 [-109.07820478472672, 36.30452158073394]]],
        scale=9000,
        crs='EPSG:4326'
    )
    task.start()

    task = ee.batch.Export.image.toDrive(
        image=out.clip(midwest_roi),
        description='mw_sm',
        folder='GEE_Exports',
        region=[[[-89.56648603472672, 39.595549379376116],
                 [-89.56648603472672, 35.019220702712595],
                 [-82.49129072222672, 35.019220702712595],
                 [-82.49129072222672, 39.595549379376116]]],
        scale=9000,
        crs='EPSG:4326'
    )
    task.start()
