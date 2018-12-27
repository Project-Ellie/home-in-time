[ -z "$WORK_ROOT" ] && echo set WORK_ROOT to a particular value of your choice and find your results in gs://going-tfx/WORK_ROOT && exit -1
export PYTHONPATH=${PYTHONPATH}:${PWD}
python -m prep.task \
    --project=going-tfx \
    --base_dir=gs://going-tfx/$WORK_ROOT/ \
    --sample_rate=1.0 \
    --prefix=atl_june \
    --encode=tfrecord \
    --runner=DataflowRunner