import tensorflow as tf

from tf_ops import sequential_group

#def model_fn(features, labels, mode, params):

"""
latent_dim,
                 wgan_lambda,
                 n_critic_steps
"""

# TODO look at export strategies.


def model_fn(generator, critic, X_sampled, mode, params):
    N = tf.placeholder_with_default(tf.constant(64), shape=[])
    Z = tf.placeholder_with_default(tf.random_normal((N, params.latent_dim), mean=0.0, stddev=1.0),
                                    shape=(None, params.latent_dim))

    with tf.variable_scope("Generator"):
        X_generated = X_fake = generator(Z)

    with tf.variable_scope("Critic"):
        C_real = critic(X_sampled)

    with tf.variable_scope("Critic", reuse=True):
        C_fake = critic(X_fake)

    with tf.variable_scope("Critic", reuse=True):
        epsilon_shape = tf.concat((tf.shape(X_fake)[:1], tf.ones_like(tf.shape(X_fake)[1:])), axis=0)
        epsilon = tf.random_uniform(shape=epsilon_shape, minval=0.0, maxval=1.0)
        X_interpolated = epsilon * X_sampled + (1.0 - epsilon) * X_fake
        C_X_interpolated_grads = tf.gradients(critic(X_interpolated), X_interpolated)[0]
        C_X_interpolated_grads_norm = tf.norm(C_X_interpolated_grads, ord=2, axis=1)
        gradient_penalty = params.wgan_lambda * tf.reduce_mean(tf.square((C_X_interpolated_grads_norm - 1.0)))

    with tf.name_scope("Training"):
        with tf.name_scope("Critic_loss"):
            # Note: Want to maximize mean output of C_real, minimize mean output of C_fake, and minimize
            # the gradient penalty.
            C_loss = -(tf.reduce_mean(C_real) - tf.reduce_mean(C_fake)) + gradient_penalty

        with tf.name_scope("Generator_loss"):
            # Note: Want to maximize mean output of C_fake.
            G_loss = -tf.reduce_mean(C_fake)

        C_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Critic")
        G_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="Generator")

        global_step = tf.train.get_or_create_global_step()

        critic_train_step_fn = lambda: (tf.train.AdamOptimizer(learning_rate=1e-4)
                                        .minimize(C_loss, global_step=None, var_list=C_variables))

        generator_train_step_fn = lambda: (tf.train.AdamOptimizer(learning_rate=1e-4)
                                           .minimize(G_loss, global_step=global_step, var_list=G_variables))

        # TODO: Not sure if this will actually be equivalent of running sess.run() separately on the ops.
        train_ops = [critic_train_step_fn] * params.n_critic_steps + [generator_train_step_fn]
        train_step = sequential_group(*train_ops)

    # TODO: predictions = X_generated I guess, loss is actually two, hmm

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=X_generated,
        loss=(C_loss, G_loss),
        train_op=train_step,
        eval_metric_ops=metrics)

