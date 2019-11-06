import ee


def data_join(left, right):

    data_filter = ee.Filter.maxDifference(
        difference=(24*60*60*1000),
        leftField='system:time_start',
        rightField='system:time_start'
    )

    filter_join = ee.Join.saveBest(
        matchKey='match',
        measureKey='delta_t'
    )

    out = ee.ImageCollection(filter_join.apply(left, right, data_filter))
    out = out.map(lambda img: img.addBands(img.get('match')))

    return out


def apply_match_proj(coll, proj):

    def match_proj(img):
        return img.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=20000
        ).reproject(crs=proj)

    return coll.map(match_proj)