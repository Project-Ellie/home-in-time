def make_input_fns():
    from train.make_csv_input_fn import make_csv_input_fn
    from train.make_tfr_input_fn import make_tfr_input_fn
    
    return {
        'csv': make_csv_input_fn,
        'tfr': make_tfr_input_fn
    }
