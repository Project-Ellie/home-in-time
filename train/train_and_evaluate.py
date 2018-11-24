def train_and_evaluate(options):

    import tensorflow as tf
    from tensorflow.estimator import RunConfig
    from tensorflow.contrib.distribute import MirroredStrategy
    import mlflow
    
    from train.make_model_fn import make_model_fn
    from train.make_tft_serving_input_fn import make_tft_serving_input_fn
    from train.create_feature_columns import create_feature_columns
    from train.make_tfr_input_fn import make_tfr_input_fn
    from train.make_hypotheses import make_hypotheses
    from train.make_input_fns import make_input_fns

    
    with mlflow.start_run():

        log_params = [
            'base_dir',
            'file_format',
            'train_batch_size',
            'max_train_steps',
            'reader_num_threads',
            'parser_num_threads',
            'prefetch_buffer_size'    
        ]
        
        for key in log_params:
            mlflow.log_param(key, options[key])

        ##################################################################
        #   Train and Eval Input Functions
        ##################################################################
        make_input_fn=make_input_fns()[options['file_format']]

        train_input_fn = make_input_fn(options['train_data_pattern'], 
                                       options['train_batch_size'],
                                       options)    

        eval_input_fn = make_input_fn(options['eval_data_pattern'], 
                                      options['eval_batch_size'],
                                      options)


        ##################################################################
        #   Create the hypothesis and the model_fn
        ##################################################################
        hypothesis = make_hypotheses()[options['hypothesis']]    
        feature_columns = create_feature_columns()
        model_fn = make_model_fn(feature_columns, options, hypothesis )


        ##################################################################
        #    Train and Eval Spec
        ##################################################################
        serving_input_fn = make_tft_serving_input_fn(options['metadata_dir'])
        exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)

        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, 
            max_steps=options['max_train_steps'])

        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn, exporters=exporter,
            steps = options['eval_steps'],
            throttle_secs=options['throttle_secs'],
            start_delay_secs=0)


        ##################################################################
        #   Create and configure the estimator
        ##################################################################
        strategy = MirroredStrategy() if options['distribute'] else None
        config = RunConfig(model_dir=options['model_dir'],
                           save_summary_steps=options['save_summary_steps'],
                           train_distribute=strategy, 
                           save_checkpoints_steps=options['save_checkpoints_steps'],
                           log_step_count_steps=options['log_step_count_steps'])

        estimator = tf.estimator.Estimator(
                config=config,
                model_fn=model_fn)


        ##################################################################
        #   Finally, train and evaluate the model
        ##################################################################
        final_eval = tf.estimator.train_and_evaluate(
            estimator, 
            train_spec=train_spec, 
            eval_spec=eval_spec)
        
        if final_eval[0] is not None:
            mlflow.log_metric('loss', final_eval[0]['loss'])
            mlflow.log_metric('mean_error', final_eval[0]['mean_error'])

        return final_eval
