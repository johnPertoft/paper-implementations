import tensorflow as tf

from generative.models import WGAN_GP
from data.mixture_of_gaussians_2d import input_tensor
from generative.models.common.tests import test_session_setup


class TestWGAN(tf.test.TestCase):
    def test_fit(self):

        X_sampled, mixture = input_tensor(64, return_mixture=True)
        wgan_gp = WGAN_GP(X_sampled,
                          latent_dim=10,
                          global_step=tf.Variable(0),
                          generator_final_activation=None)

        with self.test_session() as sess, test_session_setup(sess):

            sampled_log_prob = tf.reduce_sum(mixture.log_prob(wgan_gp.X_generated))

            for i in range(10000):
                if i % 100 == 0:
                    print(sess.run(sampled_log_prob))

                wgan_gp.train_step(sess)

            generated = sess.run(wgan_gp.X_generated)

            # TODO: Check that log probs are increasing and/or reaching some reasonable level
            # Enough to check for upwards trend in log probs?
            # Or define the model to output the parameters of the MoG instead?
