import tensorflow as tf


# TODO: remove this file
def generate(model_path, model, output_dir):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        return model.generate_results(sess, output_dir)
