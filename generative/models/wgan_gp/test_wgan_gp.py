import tensorflow as tf

from generative.models import WGAN_GP
from data.mixture_of_gaussians_2d import input_tensor
from generative.models.common import mlp
from generative.models.common.tests import test_session_setup


class TestWGAN(tf.test.TestCase):
    def test_fit(self):

        X_sampled, mixture = input_tensor(64, n_gaussians=10, stddev=0.5, return_mixture=True)

        def simple_generator(Z):
            return mlp(Z, layer_sizes=(128, 128, 2), intermediate_activation_fn=tf.nn.relu, final_activation_fn=None)

        def simple_critic(X):
            return mlp(X, layer_sizes=(128, 128, 1), intermediate_activation_fn=tf.nn.relu, final_activation_fn=None)

        wgan_gp = WGAN_GP(simple_generator,
                          simple_critic,
                          X_sampled,
                          latent_dim=10,
                          wgan_lambda=10,
                          n_critic_steps=5)

        with self.test_session() as sess, test_session_setup(sess):

            sampled_log_prob = tf.reduce_sum(mixture.log_prob(wgan_gp.X_generated))

            for i in range(5000):
                if i % 100 == 0:
                    log_prob = sess.run(sampled_log_prob)
                    print(log_prob)

                wgan_gp.train_step(sess)

            # TODO: Check that log probs are increasing and/or reaching some reasonable level
            # Enough to check for upwards trend in log probs?
            # Or define the model to output the parameters of the MoG instead?
