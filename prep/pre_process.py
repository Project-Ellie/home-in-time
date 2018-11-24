def pre_process(row):
    import tensorflow_transform as tft
    from tools import tf_haversine

    def add_engineered(row):
        dep_lat = row['DEP_LAT']
        dep_lon = row['DEP_LON']
        arr_lat = row['ARR_LAT']
        arr_lon = row['ARR_LON']

        row['DEP_HOD'] = row['DEP_T'] // 100
        row.pop('DEP_T')  # no longer needed

        row['DIFF_LAT'] = arr_lat - dep_lat
        row['DIFF_LON'] = arr_lon - dep_lon
        row['DISTANCE'] = tf_haversine(arr_lat, arr_lon, dep_lat, dep_lon)
        return row

    def scale_floats(row):
        for c in ['MEAN_TEMP_DEP', 'MEAN_VIS_DEP', 'WND_SPD_DEP', 'MEAN_TEMP_ARR', 'MEAN_VIS_ARR', 'WND_SPD_ARR', 'DEP_DELAY',
                 'DIFF_LAT', 'DIFF_LON', 'DISTANCE']:
            row[c] = tft.scale_to_0_1(row[c])
        return row

    def categorical_from_strings(row):
        row['AIRLINE'] = tft.string_to_int(row['AIRLINE'])
        row['ARR'] = tft.string_to_int(row['ARR'])
        return row
    
    row = row.copy()
    row = add_engineered(row)
    row = scale_floats(row)
    row = categorical_from_strings(row)
    return row
