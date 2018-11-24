def exec_pipeline_prod (options, train_dir, eval_dir, test_dir, 
                        metadata_dir, tmp_dir, 
                        fractions, sample_rate, prefix, 
                        encode='tfrecord', 
                        runner='DirectRunner'):
    
    import os
    import tensorflow_transform as tft
    import tensorflow_transform.beam.impl as beam_impl
    import apache_beam as beam
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import dataset_schema
    from tensorflow_transform.beam.tft_beam_io import transform_fn_io
    
    from train.model_config import (SIGNATURE_COLUMNS, TRAINING_COLUMNS,
        TRAINING_METADATA, SIGNATURE_METADATA, ORDERED_TRAINING_COLUMNS)
    from prep.pre_process import pre_process
    from prep.sample_queries import sample_queries

    with beam.Pipeline(runner, options=options) as p:
        with beam_impl.Context(temp_dir=tmp_dir):

            def write_to_files(data, prefix, phase):
                tfr_encoder = tft.coders.ExampleProtoCoder(t_metadata.schema)            
                if encode in ['tfrecord', 'both', None]:
                    _ = (data
                        | ('EncodeTFRecord_' + phase) >> beam.Map(tfr_encoder.encode)
                        | ('WriteTFRecord_' + phase) >> beam.io.WriteToTFRecord(prefix+'_tfr'))

                if encode in ['csv', 'both', None]:
                    csv_encoder = tft.coders.CsvCoder(ORDERED_TRAINING_COLUMNS, TRAINING_METADATA.schema)    
                    _ = (data 
                        | ('EncodeCSV_train' + phase) >> beam.Map(csv_encoder.encode)
                        | ('WriteText_train' + phase) >> beam.io.WriteToText(file_path_prefix=prefix+'_csv'))
        
            # Process training data and obtain transform_fn
            #
            queries = sample_queries(SIGNATURE_COLUMNS, fractions, sample_rate)

            signature_data = (p | "ReadFromBigQuery_train"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['train'], use_standard_sql=True)))
            signature_dataset = (signature_data, SIGNATURE_METADATA)
            
            tds, transform_fn = (signature_dataset | "AnalyzeAndTransform" 
                        >> beam_impl.AnalyzeAndTransformDataset(pre_process))
            t_data, t_metadata = tds

            train_prefix = os.path.join(train_dir, prefix)
            write_to_files(t_data, train_prefix, 'train')
            
            #  Process evaluation data with the obtained transform_fn
            #
            signature_data = (p | "ReadFromBigQuery_eval"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['eval'], use_standard_sql=True))) 
            signature_dataset = (signature_data, SIGNATURE_METADATA)

            t_dataset = ((signature_dataset, transform_fn) 
                         | "TransformEval" >> beam_impl.TransformDataset())
            t_data, t_metadata = t_dataset

            eval_prefix = os.path.join(eval_dir, prefix)
            write_to_files(t_data, eval_prefix, 'eval')

            #  Also process test data with the obtained transform_fn
            #
            signature_data = (p | "ReadFromBigQuery_test"  
                              >> beam.io.Read(beam.io.BigQuerySource(
                                  query=queries['test'], use_standard_sql=True)))
            signature_dataset = (signature_data, SIGNATURE_METADATA)

            t_dataset = ((signature_dataset, transform_fn) 
                         | "TransformTest" >> beam_impl.TransformDataset())
            t_data, t_metadata = t_dataset           

            test_prefix = os.path.join(test_dir, prefix)
            write_to_files(t_data, test_prefix, 'text')
            
            # save transforma function to disk for use at serving time
            #
            transform_fn | 'WriteTransformFn' >> transform_fn_io.WriteTransformFn(metadata_dir)
