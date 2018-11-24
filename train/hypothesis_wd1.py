def hypothesis_wd1(features, feature_columns, options):
    
    import tensorflow as tf

    
    wide_input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns['wide'])

    deep_input_layer = tf.feature_column.input_layer( 
        features, feature_columns=feature_columns['deep'])

    # 40 x 20
    h1 = tf.layers.dense(deep_input_layer, 20, activation='relu')
    
    # 20 x 10
    h2 = tf.layers.dense(h1, 10, activation='relu')

    # 10 x 1
    o1 = tf.layers.dense(h2, 1, activation=None)

    o2 = tf.layers.dense(wide_input_layer, 1, activation=None)
    
    o = tf.concat([o1, o2], axis=1)
    
    hypothesis = tf.layers.dense(o, 1, activation=None)

    return hypothesis
