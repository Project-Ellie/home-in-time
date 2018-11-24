def create_feature_columns():
    """
        returns: a dict of features columns for wide and deep input
    """
    
    from tensorflow.feature_column import indicator_column as ind
    from tensorflow.feature_column import numeric_column as num
    from tensorflow.feature_column import bucketized_column as buck
    from tensorflow.feature_column import crossed_column as cross
    from tensorflow.feature_column import embedding_column as emb
    from tensorflow.feature_column import categorical_column_with_identity as cid
    
    ################################################################
    #  Numerical columns for the pre-processed features
    ################################################################
    feature_columns = [
        num(col) for col in [
            'DEP_DELAY',  
            'MEAN_TEMP_DEP','MEAN_VIS_DEP','WND_SPD_DEP',
            'MEAN_TEMP_ARR','MEAN_VIS_ARR','WND_SPD_ARR',
            'DIFF_LAT','DIFF_LON','DISTANCE']]
    
    ################################################################
    #  categorical from ints, bucket counts from examination of the 
    #  full dataset
    ################################################################
    airline = ind(cid('AIRLINE', num_buckets=30))
    arrival = ind(cid('ARR', num_buckets=400))
    
    ################################################################
    #  Crossed and embedded
    ################################################################
    lat_boundaries = range(10,80,5)
    lon_boundaries = range(-100, -55, 5)
    cross_size = len(lat_boundaries) * len(lon_boundaries)

    arr_geo_emb = emb(cross([
        buck(num('ARR_LAT'), lat_boundaries), 
        buck(num('ARR_LON'), lon_boundaries)], cross_size), 10)

    dep_geo_emb = emb(cross([
        buck(num("DEP_LAT"), lat_boundaries), 
        buck(num("DEP_LON"), lon_boundaries)], cross_size), 10)

    dep_how_emb = emb(cross([
        cid("DEP_HOD", num_buckets=24), 
        cid("DEP_DOW", num_buckets=8)], 7*24), 10)

    ################################################################
    #  all together
    ################################################################
    return {
        'deep': feature_columns + [dep_how_emb, arr_geo_emb, dep_geo_emb],
        'wide': [airline, arrival]}
