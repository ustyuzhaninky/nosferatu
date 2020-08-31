import os
import tempfile
import tensorflow as tf
from unittest import TestCase
import numpy as np
from tensorflow import keras
from nosferatu.neocortex.OSAR import multiHeadAttention

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

np.random.seed(234)


class TestmultiHeadAttention(TestCase):

    def test_ops_base(self):
        """Test of the baseline functionality"""
        tf.config.experimental_run_functions_eagerly(True)
        
        features_in = 1
        features_out = 2
        timesteps = 10
        batch_size = 4
        heads = 4
        
        keys = tf.constant(np.array([[0.0], [1.0], [0.5]]), dtype=float)
        values = tf.constant(np.array([[1, 0.1], [0.1, 1], [0, 0]]), dtype=float)
        query = tf.constant(
            np.array([[0.0], [0.5], [0.0], [1.0], [0.0], [0.5]]), dtype=float)
        wq = tf.ones((heads, features_in), dtype=float)
        wk = tf.ones((heads, features_in), dtype=float)
        wv = tf.ones((heads, features_out), dtype=float)
        wh = tf.ones((heads, features_out), dtype=float)

        ans = multiHeadAttention(query, keys, values, wq, wk, wv, wh)
        print(ans)

