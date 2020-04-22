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
import tensorflow.compat.v1 as tf
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
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences

class Buffer(Layer):
    '''
    Buffer layer for RNNs injection. Works like FIFO memory.
    '''

    def __init__(self, n_steps:int, max_shift:int=None, initializer=initializers.glorot_uniform, **kwargs):
        '''

        
        @param: n_steps, int, a number of timesteps of the input vector to be saved in the buffer
        (should be bigger than batch dimension or data all extra data will be lost form the batch)
        
        @param: max_shift, int, a value of maximum possible shift of the buffer for one iteration.
        Any values beyond the scope will be lost.

         @param: iniializer, a buffer initializer. Standart is white noise from `glorot_uniform`.

        '''
        super(Buffer, self).__init__(trainable=False, **kwargs)
        # if max_shift and max_shift >= n_steps:
        #     raise ValueError('Maximum shift must smaller than number of steps in the buffer: `max_shift > n_steps`.')

        self.n_steps = n_steps
        self.max_shift = max_shift
        self.initializer = initializer
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `Buffer` '
                            'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        self.input_spec = InputSpec(min_ndim=2,
                                axes={-1: last_dim})

        self.buffer = self.add_weight(name='buffer', shape=(self.n_steps,) + input_shape[1:], 
                                        initializer=self.initializer, trainable=False)
        
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
    
    def call(self, inputs): #it's

        self.add_update(tf.assign(self.buffer, self._buffer_update(inputs)))

        out = tf.reshape(self.buffer, (inputs.shape[0], int(self.n_steps / inputs.shape[0]),) + self.buffer.shape[1:])

        return out
    
    def _buffer_update(self, inputs):
        shape = inputs.shape
        if shape[0] > self.n_steps:
            raise ValueError(
                    'The number of steps in the buffer cannot be smaller than the batch dimension.'
                )

        masked_inputs =  tf.keras.layers.concatenate([inputs, tf.zeros((self.n_steps-shape[0],)+shape[1:])], axis=0)
        nullifying_mask = tf.keras.layers.concatenate([tf.zeros(shape), tf.ones((self.n_steps-shape[0],)+shape[1:])], axis=0)
        
        rolled = tf.roll(self.buffer, shift=shape[0], axis=0)
        out = rolled * nullifying_mask + masked_inputs

        return out

    def get_config(self):
        config = {
            'n_steps': self.n_steps,
            'max_shift': self.max_shift,
        }
        base_config = super(Buffer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# #python -um nosferatu.sequence.seq_layers
# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
# inputs = tf.keras.Input(shape=(2), batch_size=1)
# outputs = Buffer(5)(inputs)
# model = tf.keras.Model(inputs, outputs)

# model.compile(optimizer='adam', loss='mse')
# model.summary()
# x = np.random.randn(100, 2)

# for i in range(3):
#     print(model.predict(x[i:(2*i+1)]))
# # print(model.predict(x[:2]))
