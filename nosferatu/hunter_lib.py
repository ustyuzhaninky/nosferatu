# coding=utf-8
# Copyright 2020 Konstantin Ustyuzhanin.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math


import gin
import gym
from gym.spaces.box import Box
import numpy as np
import tensorflow as tf
import tensorflow as tfe
from tensorflow.keras import backend as K

import cv2
from tensorflow.keras import layers as layers
from nosferatu.capsules import cap_layers
from nosferatu.sequence import seq_layers
from nosferatu.neocortex import Online
from nosferatu.dopamine.discrete_domains.atari_lib import *

class RainbowCapsuleHunter(tf.keras.Model):
  """The classic CapsNet used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, num_capsule, dim_capsule, routings=3, share_weights=True, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      num_capsule: int, the nnumber of capsules in the capsule layer
      dim_capsule: int, the dimension of each capsule
      routings: int, the number of rooting operations
      share_weights: bool, shows whether the weights should be shared between capsules (leave it be)
      name: str, used to crete scope for network parameters.
    """
    super(RainbowCapsuleHunter, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.

    # self.conv1 = tf.keras.layers.Conv2D(
    #     64, [3, 3], strides=4, padding='same', activation=activation_fn,
    #     kernel_initializer=self.kernel_initializer, name='Conv')
    # self.conv2 = tf.keras.layers.Conv2D(
    #     64, [3, 3], strides=2, padding='same', activation=activation_fn,
    #     kernel_initializer=self.kernel_initializer, name='Conv')
    # self.avg1 = tf.keras.layers.AveragePooling2D((2,2), name='AvgPooling')
    # self.conv3 = tf.keras.layers.Conv2D(
    #     128, [3, 3], strides=1, padding='same', activation=activation_fn,
    #     kernel_initializer=self.kernel_initializer, name='Conv')
    self.conv4 = tf.keras.layers.Conv2D(
        128, [3, 3], strides=1, padding='same', activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='Conv')
    self.reshape1 = tf.keras.layers.Reshape((-1, 128), name='Reshape')
    self.primary_caps = cap_layers.Capsule(8, 3, 3, True)
    self.digit_caps =  tf.keras.layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))

    # self.flatten = tf.keras.layers.Flatten()
    # self.reshape2 = tf.keras.layers.Reshape((128, 100), name='Reshape')
    # self.buffer = seq_layers.Buffer(128)
    
    # self.lstm1 = tf.keras.layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True,
    #                                   kernel_initializer=self.kernel_initializer)
    # self.lstm2 = tf.keras.layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = True,
    #                                   kernel_initializer=self.kernel_initializer)
    # self.lstm3 = tf.keras.layers.LSTM(128, dropout = 0.2, recurrent_dropout = 0.2, return_sequences = False,
    #                                   kernel_initializer=self.kernel_initializer)
    
    # self.conv3 = tf.keras.layers.Conv2D(
    #     64, [3, 3], strides=1, padding='same', activation=activation_fn,
    #     kernel_initializer=self.kernel_initializer, name='Conv')
    # self.flatten = tf.keras.layers.Flatten()
    self.bolzmann1 = Online.OnlineBolzmannCell(512, online=True)
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.bolzmann3 = Online.OnlineBolzmannCell(10, online=True)
    self.dropout3 = tf.keras.layers.Dropout(0.2)
    self.bolzmann4 = Online.OnlineBolzmannCell(512, online=True)
    self.dropout4 = tf.keras.layers.Dropout(0.2)
    # self.dense1 = tf.keras.layers.Dense(
        # 512, activation=activation_fn,
        # kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')
    
    self.noise = tf.keras.layers.GaussianNoise(0.1)

  def call(self, state):
    """Creates the output tensor/op given the state tensor as input.

    See https://www.tensorflow.org/api_docs/python/tf/keras/Model for more
    information on this. Note that tf.keras.Model implements `call` which is
    wrapped by `__call__` function by tf.keras.Model.

    Args:
      state: Tensor, input tensor.
    Returns:
      collections.namedtuple, output ops (graph mode) or output tensors (eager).
    """

    # print(state.shape)
    x = tf.cast(state, tf.float32)
    x = x / 255.
    
    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.avg1(x)
    # x = self.conv3(x)
    x = self.conv4(x)
    x = self.reshape1(x)
    x = self.primary_caps(x)
    x = self.digit_caps(x)

    # x = self.flatten(x)
    # x = self.reshape2(x)
    # x = self.permute(x)
    # x = self.buffer(x)
    # x = self.lstm1(x)
    # x = self.lstm2(x)
    # x = self.lstm3(x)
    x = self.bolzmann1(x)
    x = self.dropout1(x)
    x = self.bolzmann2(x)
    x = self.dropout2(x)
    x = self.bolzmann3(x)
    x = self.dropout3(x)
    x = self.bolzmann4(x)
    x = self.dropout4(x)
    x = self.bolzmann5(x)
    x = self.dropout5(x)
    # x = self.dense1(x)
    x = self.dense2(x)
    x = self.noise(x, training=True)

    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)
