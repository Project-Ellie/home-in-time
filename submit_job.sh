
    JOBNAME=homeintime_$(date -u +%y%m%d_%H%M%S)
    gcloud ml-engine jobs submit training $JOBNAME \
    --region=us-east1 \
    --module-name=train.task \
    --package-path=train \
    --job-dir=gs://going-tfx/jobs \
    --staging-bucket=gs://going-tfx \
    --scale-tier=STANDARD_1 \
    --runtime-version=1.8 \
    -- \
    --eval_steps="10"  \
    --parser_num_threads="16"  \
    --eval_data_pattern="gs://going-tfx/samples/eval_data/atl_june_tfr*"  \
    --train_batch_size="256"  \
    --shuffle_buffer_size="10000"  \
    --eval_batch_size="1024"  \
    --sloppy_ordering="True"  \
    --reader_num_threads="16"  \
    --file_format="tfr"  \
    --log_step_count_steps="200"  \
    --model_dir="gs://going-tfx/samples/model"  \
    --throttle_secs="30"  \
    --optimizer="sgd"  \
    --learning_rate="0.001"  \
    --hypothesis="linear"  \
    --save_summary_steps="100"  \
    --max_train_steps="8000"  \
    --prefetch_buffer_size="10000"  \
    --metadata_dir="gs://going-tfx/samples/metadata"  \
    --train_data_pattern="gs://going-tfx/samples/train_data/atl_june_tfr*"  \
    --save_checkpoints_steps="2000"  \
    --base_dir="gs://going-tfx/samples"  \

    