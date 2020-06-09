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
#
# author: Konstantin Ustyuzhanin
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.python.framework import auto_control_deps
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.framework import errors

class BufferCell(Layer):
    '''
    Simple buffer cell for RNN/LSTM injection. Works like FIFO memory.
    Returns buffer
    
    '''
    def __init__(self, n_steps, **kwargs):
          self.n_steps = n_steps
          
          self._buffer = None
          super(BufferCell, self).__init__(**kwargs)

    @property
    def buffer(self):
        return self._buffer
    
    @buffer.setter
    @trackable.no_automatic_dependency_tracking
    def buffer(self, buffer):
        self._buffer = buffer

    def build(self, input_shape):
          # self._buffer = self.add_weight(
          #     name='buffer',
          #     shape=(self.n_steps,) + input_shape[1:], 
          #     initializer='zeros', trainable=False)
          self.built = True
      
    def call(self, inputs):
        input_shape = inputs.shape
        
        # if input_shape[0] > self.n_steps:
        #     raise ValueError(
        #             'The number of steps in the buffer cannot be smaller than the batch dimension.'
        #         )
        if self.n_steps <= input_shape[0]:
            d_mask = self.n_steps
            n_mask = input_shape[0] - d_mask
            saved_inps = tf.convert_to_tensor(inputs[-d_mask:])
            if len(saved_inps.shape) < 2:
                saved_inps = tf.reshape(saved_inps, (d_mask)+saved_inps.shape)
            masked_inputs = saved_inps
            nullifying_mask = 0
        else:
            d_mask = input_shape[0]
            n_mask = self.n_steps - input_shape[0]
            saved_inps = inputs
            if len(saved_inps.shape) < 2:
                saved_inps = tf.reshape(saved_inps, (d_mask)+saved_inps.shape)
            masked_inputs = tf.keras.layers.concatenate([saved_inps, tf.zeros(
                (n_mask,)+saved_inps.shape[1:], dtype=inputs.dtype)], axis=0)
            nullifying_mask = tf.keras.layers.concatenate([tf.zeros((d_mask,)+saved_inps.shape[1:], dtype=inputs.dtype), tf.ones(
                (n_mask,)+saved_inps.shape[1:], dtype=inputs.dtype)], axis=0)
        
        if self.buffer is not None:
          rolled = tf.roll(self.buffer, shift=d_mask, axis=0)
          updated_buffer = rolled * nullifying_mask + masked_inputs
        else:
          self.buffer = self.add_weight(
              name='buffer',
              shape=(self.n_steps,) + input_shape[1:],
              initializer='zeros', trainable=False, dtype=inputs.dtype)
          updated_buffer = self.buffer * nullifying_mask + masked_inputs
        
        if inputs.shape[0] < self.n_steps:
            batch_dim = 1
        else:
            batch_dim = int(inputs.shape[0] / self.n_steps)
        output = tf.reshape(updated_buffer, (batch_dim, self.n_steps,) + updated_buffer.shape[1:])
        
        if self.buffer is not None:
          self.buffer.assign(updated_buffer)
        else:
          self.buffer = updated_buffer
        return output
    
    def get_config(self):
        config = {
          'n_steps': self.n_steps,
        }

        base_config = super(BufferCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Buffer(Layer):
    '''
    Buffer layer for RNNs injection. Works like FIFO memory.
    '''

    def __init__(self, n_steps:int, **kwargs):
        '''

        @param: n_steps, int, a number of timesteps of the input vector to be saved in the buffer
        (should be bigger than batch dimension or data all extra data will be lost form the batch)
        
        @param: max_shift, int, a value of maximum possible shift of the buffer for one iteration.
        Any values beyond the scope will be lost.

        @param: iniializer, a buffer initializer. Standart is white noise from `glorot_uniform`.

        '''
        super(Buffer, self).__init__(trainable=False, dynamic=False, **kwargs)

        self.n_steps = n_steps
        self.cell = BufferCell(self.n_steps)
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Buffer` '
                            'should be defined. Found `None`.')
        if isinstance(input_shape, list):
            input_shape = input_shape[0]
            self.cell.build(input_shape)
        
        self.built = True
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_shape = input_shape.with_rank_at_least(2)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)
        return (input_shape[0], self.n_steps) + input_shape[1:]
    
    def call(self, inputs):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = K.convert_inputs_if_ragged(inputs)

        def step(inputs):
            output = self.cell.call(
                inputs)
            return output

        output = step(inputs)

        return output
    
    def get_config(self):
        cell = {
          'class_name': self.cell.__class__.__name__,
          'config': self.cell.get_config()
        }
        config = {'cell': cell}
        base_config = super(Buffer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from tensorflow.python.keras.layers import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'))
        return cls(cell, **config)


# python -um nosferatu.sequence.seq_layers
# tf.config.experimental_run_functions_eagerly(True)

# inputs = tf.keras.Input(shape=(2,), batch_size=1)
# buff = Buffer(5)
# outputs = buff(inputs)
# model = tf.keras.Model(inputs, outputs)

# model.compile(optimizer='adam', loss='mse')
# # model.summary()
# x = np.random.randn(100, 2)

# for i in range(5):
#     model.predict([[i, i]])
# assert buff.cell.buffer.numpy().all() == np.array([[4-i, 4-i] for i in range(5)]).all()
