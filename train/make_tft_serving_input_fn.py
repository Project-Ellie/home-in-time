def make_tft_serving_input_fn(metadata_dir):

    import tensorflow as tf
    import tensorflow_transform as tft
    from train.model_config import SIGNATURE_INT_COLUMNS
    from train.model_config import SIGNATURE_FLOAT_COLUMNS
    from train.model_config import SIGNATURE_STR_COLUMNS
        
    def _input_fn():
        # placeholders for all the raw inputs
        placeholders = {
            key: tf.placeholder(name = key, shape=[None], dtype=tf.int64)
            for key in SIGNATURE_INT_COLUMNS
        }
        placeholders.update({
            key: tf.placeholder(name = key, shape=[None], dtype=tf.float32)
            for key in SIGNATURE_FLOAT_COLUMNS
        })

        placeholders.update({
            key: tf.placeholder(name = key, shape=[None], dtype=tf.string)
            for key in SIGNATURE_STR_COLUMNS
        })

        # transform using the saved model in transform_fn        
        transform_output = tft.TFTransformOutput(transform_output_dir=metadata_dir)
        features = transform_output.transform_raw_features(placeholders)
            
        return tf.estimator.export.ServingInputReceiver(features, placeholders)

    return _input_fn
