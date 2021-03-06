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
import matplotlib.pyplot as plt
from absl import logging

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
from nosferatu.sequence import *
from nosferatu.neocortex import Online
from nosferatu.dopamine.discrete_domains.atari_lib import *
from nosferatu.keras_segmentation.pretrained import pspnet_101_voc12
from nosferatu.keras_segmentation.models.unet import vgg_unet

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

    # vgg19
    self.vgg19 = tf.keras.applications.VGG19(
          input_shape=(84, 84, 3),
          include_top=False,
          weights="imagenet",
          input_tensor=None,
          pooling='max',
          classes=1000,
      )
    self.vgg19.trainable = False
    self.vgg19.compile(optimizer='rmsprop', loss='mse')
    self.vgg19.summary()
    
    # basic recognition part
    # self.conv1 = tf.keras.layers.Conv2D(
    #         64, [3, 3], padding='same', activation='relu',
    #         kernel_initializer=self.kernel_initializer, name='Conv1')
    # self.conv2 = tf.keras.layers.Conv2D(
    #     64, [3, 3],padding='same', activation='relu',
    #     kernel_initializer=self.kernel_initializer, name='Conv2')
    # self.avgpool =tf.keras.layers.AveragePooling2D((2, 2), name='avgpool')
    # self.conv3 = tf.keras.layers.Conv2D(
    #     128, [3, 3], padding='same', activation='relu',
    #     kernel_initializer=self.kernel_initializer, name='Conv3')
    # self.conv4 = tf.keras.layers.Conv2D(
    #     128, [3, 3], padding='same', activation='relu',
    #     kernel_initializer=self.kernel_initializer, name='Conv4')
    self.reshape1 = tf.keras.layers.Reshape((-1, 128), name='Reshape')

    # Capsule layers
    self.primary_caps = cap_layers.Capsule(10, 16, 3, True)
    # self.secondary_caps = cap_layers.Capsule1D(8, 3, 3, True)
    self.digit_caps = tf.keras.layers.Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))
    # self.reshape1 = tf.keras.layers.Reshape((24, -1), name='Reshape')
    
    # One sencond buffer storage
    self.buf = Buffer(24)
    self.gru1 = tf.keras.layers.GRU(24, use_bias=True, 
                                    recurrent_dropout=0.2,
                                    dropout=0.2, return_sequences=False,)
    
    

    ### My custom caps
    # self.conv4 = tf.keras.layers.Conv2D(
    #     128, [3, 3], strides=1, padding='same', activation=activation_fn,
    #     kernel_initializer=self.kernel_initializer, name='Conv')
    # self.reshape1 = tf.keras.layers.Reshape((-1, 128), name='Reshape')
    # self.primary_caps = cap_layers.Capsule(8, 3, 3, True)
    # self.digit_caps =  tf.keras.layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)), output_shape=(10,))
    self.bolzmann1 = Online.OnlineBolzmannCell(512, online=True)
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.bolzmann2 = Online.OnlineBolzmannCell(10, online=True)
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    self.bolzmann3 = Online.OnlineBolzmannCell(512, online=True)
    self.dropout3 = tf.keras.layers.Dropout(0.2)
    # # self.dense1 = tf.keras.layers.Dense(
    #     # 512, activation=activation_fn,
    #     # kernel_initializer=self.kernel_initializer, name='fully_connected')
    # self.dense2 = tf.keras.layers.Dense(
    #     num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
    #     name='fully_connected')
    
    # self.noise = tf.keras.layers.GaussianNoise(0.01)

    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

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
    
    

    # cognition part
    x = self.vgg19(x)

    # x = self.conv1(x)
    # x = self.conv2(x)
    # x = self.avgpool(x)
    # x = self.conv3(x)
    # x = self.conv4(x)
    x = self.reshape1(x)

    # caps part
    x = self.primary_caps(x)
    x = self.digit_caps(x)

    # recurrent part
    x = self.buf(x)
    x = self.gru1(x)

    x = self.bolzmann1(x)
    x = self.dropout1(x)
    x = self.bolzmann2(x)
    x = self.dropout2(x)
    x = self.bolzmann3(x)
    x = self.dropout3(x)

    # hat
    x = self.dense1(x)
    x = self.dense2(x) 

    ### My custom caps
    # x = self.conv4(x)
    # x = self.reshape1(x)
    # x = self.primary_caps(x)
    # x = self.digit_caps(x)
    # x = self.bolzmann1(x)
    # x = self.dropout1(x)
    # x = self.bolzmann2(x)
    # x = self.dropout2(x)
    # x = self.bolzmann3(x)
    # x = self.dropout3(x)
    # # x = self.dense1(x)
    # x = self.dense2(x)

    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)
    return RainbowNetworkType(q_values, logits, probabilities)


class NsHunter1(tf.keras.Model):
  """The custom NsHunter used to compute agent's return distributions."""

  def __init__(self, num_actions, num_atoms, support, share_weights=True, name=None):
    """Creates the layers used calculating return distributions.

    Args:
      num_actions: int, number of actions.
      num_atoms: int, the number of buckets of the value function distribution.
      support: tf.linspace, the support of the Q-value distribution.
      share_weights: bool, shows whether the weights should be shared between capsules (leave it be)
      name: str, used to crete scope for network parameters.
    """
    super(NsHunter1, self).__init__(name=name)
    activation_fn = tf.keras.activations.relu
    self.num_actions = num_actions
    self.num_atoms = num_atoms
    self.support = support
    self.kernel_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
    # Defining layers.

    # basic recognition part
    self.conv1 = tf.keras.layers.Conv2D(
            64, [3, 3], padding='same', activation='relu',
            kernel_initializer=self.kernel_initializer, name='Conv1')
    self.conv2 = tf.keras.layers.Conv2D(
        64, [3, 3],padding='same', activation='relu',
        kernel_initializer=self.kernel_initializer, name='Conv2')
    self.avgpool =tf.keras.layers.AveragePooling2D((2, 2), name='avgpool')
    self.conv3 = tf.keras.layers.Conv2D(
        128, [3, 3], padding='same', activation='relu',
        kernel_initializer=self.kernel_initializer, name='Conv3')
    self.conv4 = tf.keras.layers.Conv2D(
        128, [3, 3], padding='same', activation='relu',
        kernel_initializer=self.kernel_initializer, name='Conv4')
    self.reshape2 = tf.keras.layers.Reshape((-1, 128), name='Reshape')

    # Capsule layers
    self.primary_caps = cap_layers.Capsule(10, 16, 3, True)
    self.digit_caps = tf.keras.layers.Lambda(
        lambda z: K.sqrt(K.sum(K.square(z), 2)))

    # Short-term bi-memory
    self.buf = Buffer(24)

    self.gru2 = tf.keras.layers.GRU(100, use_bias=True,
                                    recurrent_dropout=0.2,
                                    dropout=0.2, return_sequences=False,
                                    go_backwards=False)

    # Attention

    query_input = tf.keras.Input(shape=(None, 100), dtype='float32')
    value_input = tf.keras.Input(shape=(None, 100), dtype='float32')

    encoder = tf.keras.layers.Conv1D(
        filters=100,
        kernel_size=4,
        padding='same')
    ## Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = encoder(query_input)
    ## Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = encoder(value_input)
    ## Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = tf.keras.layers.Attention()(
        [query_seq_encoding, value_seq_encoding])

    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
        query_seq_encoding)
    query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
        query_value_attention_seq)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

    self.attention = tf.keras.Model(
        inputs=[query_input, value_input], outputs=input_layer)
    
    self.attention.summary()
    
    ### My custom caps
    self.bolzmann1 = Online.OnlineBolzmannCell(100, online=True)
    self.dropout1 = tf.keras.layers.Dropout(0.2)
    self.bolzmann2 = Online.OnlineBolzmannCell(100, online=True)
    self.dropout2 = tf.keras.layers.Dropout(0.2)
    self.bolzmann3 = Online.OnlineBolzmannCell(100, online=True)
    self.dropout3 = tf.keras.layers.Dropout(0.2)

    self.dense1 = tf.keras.layers.Dense(
        512, activation=activation_fn,
        kernel_initializer=self.kernel_initializer, name='fully_connected')
    self.dense2 = tf.keras.layers.Dense(
        num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
        name='fully_connected')

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

    x = tf.cast(state, tf.float32)
    x = x / 255.

    x = self.conv1(x)
    x = self.conv2(x)
    x = self.avgpool(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.reshape2(x)
    filters = x

    # caps part
    x = self.primary_caps(x)
    x = self.digit_caps(x)

    # recurrent part
    x = self.buf(x)
    # x = self.short_mem(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = self.gru2(x)
    # x = tf.keras.layers.BatchNormalization()(x)

    # values = tf.keras.layers.Flatten()(filters)
    # values = Buffer(100)(values)

    # x = self.attention(x, values)

    x = self.bolzmann1(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = self.dropout1(x)
    x = self.bolzmann2(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = self.dropout2(x)
    x = self.bolzmann3(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = self.dropout3(x)

    # hat
    x = self.dense1(x)
    x = self.dense2(x)

    logits = tf.reshape(x, [-1, self.num_actions, self.num_atoms])
    probabilities = tf.keras.activations.softmax(logits)
    q_values = tf.reduce_sum(self.support * probabilities, axis=2)

    tf.keras.Model(inputs=state, outputs=q_values).summary(print_fn=logging.info)

    return RainbowNetworkType(q_values, logits, probabilities)
