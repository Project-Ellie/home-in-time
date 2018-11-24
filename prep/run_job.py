def run_job(args):
    
    import datetime
    import apache_beam as beam
    from prep.exec_pipeline_prod import exec_pipeline_prod
    
    job_name = 'tft-tutorial' + '-' + datetime.datetime.now().strftime('%y%m%d-%H%M%S')    
    
    options = {
        'staging_location': args['stage_dir'],
        'temp_location': args['tmp_dir'],
        'job_name': job_name,
        'project': args['project'],
        'max_num_workers': int(args['max_workers']),
        'teardown_policy': 'TEARDOWN_ALWAYS',
        'no_save_main_session': True,
        'requirements_file': 'dataflow_requirements.txt'
    }    
    opts = beam.pipeline.PipelineOptions(flags=[], **options)

    fractions = [int(n) for n in args['fractions'].split(",")]

    exec_pipeline_prod (opts, args['train_dir'], args['eval_dir'],args['test_dir'],
                        args['metadata_dir'], args['tmp_dir'],
                        fractions, float(args['sample_rate']), args['prefix'],
                        encode=args['encode'], runner=args['runner'])
