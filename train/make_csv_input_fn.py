def make_csv_input_fn(filename_pattern, batch_size, options): 

    import tensorflow as tf
    from train.model_config import ORDERED_TRAINING_DEFAULTS
    from train.model_config import ORDERED_TRAINING_COLUMNS
    from train.model_config import LABEL_COLUMN
    
    
    def _input_fn():
        filenames = tf.gfile.Glob(filename_pattern)
        dataset = tf.data.TextLineDataset(filenames)

        def decode_csv(row):
            cols = tf.decode_csv(row, record_defaults=ORDERED_TRAINING_DEFAULTS)
            features = dict(zip(ORDERED_TRAINING_COLUMNS, cols))
            return features

        def pop_target(features):
            target = features.pop(LABEL_COLUMN)
            return features, target
        
        if options['shuffle_buffer_size'] is not None:
            dataset = dataset.shuffle(buffer_size=options['shuffle_buffer_size'])
                
        dataset = (dataset.repeat()
                   .map(decode_csv)
                   .map(pop_target)
                   .batch(batch_size))
        
        if options['distribute']:
            return dataset 
        else:
            return dataset.make_one_shot_iterator().get_next()
    
    return _input_fn
