def make_tfr_input_fn(filename_pattern, batch_size, options):
    
    import tensorflow as tf
    from train.model_config import LABEL_COLUMN
    from train.model_config import TRAINING_METADATA

    feature_spec = TRAINING_METADATA.schema.as_feature_spec()

    def _input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=filename_pattern,
            batch_size=batch_size,
            features=feature_spec,
            shuffle_buffer_size=options['shuffle_buffer_size'],
            prefetch_buffer_size=options['prefetch_buffer_size'],
            reader_num_threads=options['reader_num_threads'],
            parser_num_threads=options['parser_num_threads'],
            sloppy_ordering=options['sloppy_ordering'],
            label_key=LABEL_COLUMN)

        if options['distribute']:
            return dataset 
        else:
            return dataset.make_one_shot_iterator().get_next()
    return _input_fn
