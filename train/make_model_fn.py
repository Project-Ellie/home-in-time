def make_model_fn(feature_columns, options, hypothesis):
    
    import tensorflow as tf
    from train.make_hypotheses import make_hypotheses
    
    optimizers={
        "sgd": tf.train.GradientDescentOptimizer(learning_rate=options['learning_rate']),
        "adam": tf.train.AdamOptimizer(learning_rate=options['learning_rate']),
        "adagrad": tf.train.AdagradOptimizer(learning_rate=options['learning_rate'])
    }
    
    def _model_fn(features, labels, mode):

        out = hypothesis(features, feature_columns, options)

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=out)


        labels = tf.expand_dims(labels, -1)
        loss = tf.losses.mean_squared_error(labels, out)
        mean_error=tf.metrics.mean(tf.abs(labels-out))

        if mode == tf.estimator.ModeKeys.EVAL:    
            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss = loss,
                eval_metric_ops={'mean_error': mean_error}
            )

        else:
            optimizer = optimizers[options['optimizer']]
            train_op = optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())

            grads = optimizer.compute_gradients(loss)
            for g in grads:
                name = "%s-grad" % g[1].name
                name = name.replace(":", "_")
                tf.summary.histogram(name, g[0])
            
            return tf.estimator.EstimatorSpec(  
                mode,
                loss = loss,
                train_op = train_op)
        
    return _model_fn
