def join_paths(args):
    """
    A little helper that allows us to specify directories with the help of a particular base path
    """
    import os
    def combine_path(basedir, other):
        if other.startswith("/") or other.startswith("gs://"):
            return other
        else:
            return os.path.join(basedir, other)
    for key in ['metadata_dir', 'model_dir', 'eval_data_pattern', 'train_data_pattern']:
        args[key] = combine_path(args['base_dir'], args[key])
    return args


def variable_summaries(common_name, var):
    import tensorflow as tf
    with tf.name_scope(common_name):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stdev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stdev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)


def weight_summary (dense):
    weight = dense.op.inputs[0].op.inputs[1].op.inputs[0]
    return variable_summaries("kernel", weight)

