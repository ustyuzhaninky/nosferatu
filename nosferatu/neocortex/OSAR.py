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

import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.eager import context
from tensorflow import nn
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
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
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.framework import errors
from tensorflow.keras import activations
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from nosferatu.sequence.scale import Scale
from nosferatu.sequence.memory import Memory
from nosferatu.sequence.pos_embed import PositionalEmbedding
from nosferatu.sequence.rel_bias import RelativeBias
from nosferatu.sequence.rel_multi_head import RelativePartialMultiHeadSelfAttention

__all__ = ['OSAR', 'TransformerShift']


class TransformerShift(Layer):
    """Compressive Transformer left-shift layer with fixed short and convolution memories
       
       Layer has trainable compression function if method=`conv` is selected.
    
    """

    def __init__(self,
                 short_units,
                 conv_units,
                 method='conv',
                 compression_rate=2,
                 filters=32,
                 use_bias=True,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Compressive Transformer left-shift layer with fixed short and convolution memories

        # Arguments
            feature_units: int > 0, Number of short-term memory units in the layer.
            conv_units: int > 0, Number of convolution memory units in the layer.
            method: str [`conv`, `mean`, `max`], Defines a memory compression function (default - conv).
                Available compression functions:
                - `conv`: 1D convolution over memory sequence.
                - `mean`: 1D Mean keras.layers.AveragePooling1D.
                - `max`: 1D Max Pooling via keras.MaxPooling1D.
            compression_rate: int >= 1. Compression rate for the selected compression method.

            Only for method=`conv`:

            filters (default - 32): int > 0, Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
            use_bias (default - True): Boolean, whether the layer uses a bias vector.
            activation (default - None): Activation functon applied to the output of the compression function ( see keras.activations).
            kernel_initializer (default - 'glorot_uniform'): Initializer for the kernel weights matrix (see keras.initializers).
            bias_initializer: Initializer for the bias vector (see keras.initializers) (default - 'zeros').
            kernel_regularizer (default - None): Regularizer function applied to the kernel weights matrix (see keras.regularizers).
            bias_regularizer (default - None): Regularizer function applied to the bias vector ( see keras.regularizers).
            kernel_constraint (default - None): Constraint function applied to the kernel matrix ( see keras.constraints).
            bias_constraint (default - None): Constraint function applied to the bias vector ( see keras.constraints).

        # Inputs Shape
            +2D tensor with shape: `(batch_size, ..., features)`,
            +2D tensor with shape: `(conv_units+short_units, ..., features)`.
        # Output Shape
            +3D tensor with shape: `(conv_units+short_units, ..., features)`.
        # References
            - [Compressive-Transformer](https://arxiv.org/abs/1911.05507)

        """

        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self._return_runtime = kwargs.pop('return_runtime', False)
        # Overriding system preset for eager execution at all times
        kwargs.pop('dynamic', False)
        super(TransformerShift, self).__init__(
            dynamic=True,
            **kwargs)

        self.conv_units = conv_units
        self.short_units = short_units
        self.filters = filters
        if method not in ['conv', 'mean', 'max']:
            raise ValueError(
                f'Compression method `{method}` is not supported.')
        self.method = method
        if compression_rate < 1:
            raise ValueError('Compression rate cannot be less than 1.')
        self.compression_rate = compression_rate
        if method in ['max', 'mean']:
            self.compression_rate = np.int(np.floor(self.compression_rate))

        self.use_bias = use_bias
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 2:
            raise ValueError('The dimensions of the inputs to `TransformerShift` '
                             f'should be two tensors. Found `{input_shape}`.')
        if len(input_shape[0]) <= 2:
            raise ValueError('The dimensions of the first tensor of the inputs to `TransformerShift` '
                             f'should be at least 3D. Found `{input_shape[0]}`.')
        if len(input_shape[-1]) <= 2:
            raise ValueError('The dimensions of the second tensor of the inputs (memory) to `TransformerShift` '
                             f'should be at least 3D. Found `{input_shape[-1]}`.')
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        vector_shape = tensor_shape.TensorShape(input_shape[0])
        memory_shape = tensor_shape.TensorShape(input_shape[1])
        if tensor_shape.dimension_value(vector_shape[-1]) is None:
            raise ValueError('The last dimension of the first tensor of the inputs to `TransformerShift` '
                             'should be defined. Found `None`.')
        if tensor_shape.dimension_value(memory_shape[-1]) is None:
            raise ValueError('The last dimension of the second tensor of the inputs (memory) to `TransformerShift` '
                             'should be defined. Found `None`.')
        if tensor_shape.dimension_value(vector_shape[-1]) is not \
                tensor_shape.dimension_value(memory_shape[-1]):
            raise ValueError('The last dimension of both input tensors to `TransformerShift` '
                             f'should match. Found `{tensor_shape.dimension_value(vector_shape[-1])}` and '
                             f'`{tensor_shape.dimension_value(memory_shape[-1])}`.')
        if self.method is 'conv':
            self.compression = nn.conv1d
            self.kernel = self.add_weight(
                name='compression_kernel',
                shape=(self.compression_rate, vector_shape[-1], self.filters),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                trainable=True)
            if self.use_bias:
                self.bias = self.add_weight(
                    name='compression_bias',
                    shape=(self.filters,),
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint,
                    trainable=True)
        elif self.method is 'mean':
            self.compression = nn.avg_pool1d
        elif self.method is 'max':
            self.compression = nn.max_pool1d
        else:
            self.compression = nn.conv1d

        self.built = True

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    def call(self, inputs):
        if len(inputs) != 2:
            raise ValueError('The dimensions of the inputs to `TransformerShift` '
                             f'should be two tensors. Found `{inputs.shape}`.')
        new_tensor_input = inputs[0]
        memory_input = inputs[1]
        if len(new_tensor_input.shape) <= 2:
            raise ValueError(
                'The dimension of the input vector'
                ' should be at least 3D: `(batch_size, ..., features)`')
        if len(memory_input.shape) < 3:
            raise ValueError(
                'The dimension of the memory vector'
                ' should be at least 3D: `(conv_units+short_units, ..., features)`')
        conv_mem = memory_input[:self.conv_units]
        short_mem = memory_input[self.conv_units:]
        d_mask = new_tensor_input.shape[0]
        # Forgetting the oldest
        old_mem = tf.reshape(
            short_mem[:d_mask, ...], (d_mask,)+short_mem.shape[1:])
        # Compressing the oldest memories

        if self.method is 'conv':
            compression = nn.conv1d(old_mem,
                                    filters=self.kernel,
                                    stride=self.compression_rate,
                                    padding='SAME',
                                    name='Compression')
            if self.use_bias:
                compression = nn.bias_add(
                    compression, self.bias)
        elif self.method is 'mean':
            compression = nn.avg_pool1d(old_mem, self.compression_rate,
                                        strides=self.compression_rate, padding='same', name='Compression')
        elif self.method is 'max':
            compression = nn.max_pool1d(old_mem, self.compression_rate,
                                        strides=self.compression_rate, padding='same', name='Compression')
        else:
            compression = nn.avg_pool1d(old_mem, self.compression_rate,
                                        strides=self.compression_rate, padding='same', name='Compression')
        if self.activation is not None:
            compression = self.activation(compression)
        new_convs = tf.reshape(compression, (d_mask,)+short_mem.shape[1:])

        # Update Short-Term Memory
        short_mem = K.concatenate(
            [short_mem[:-d_mask, ...], new_tensor_input], axis=0)
        # Update Compressed Memory
        conv_mem = K.concatenate(
            [conv_mem, new_convs], axis=0)[d_mask:, ...]

        return K.concatenate([conv_mem, short_mem], axis=0)

    def get_config(self):
        config = {
            'conv_units': self.conv_units,
            'short_units': self.short_units,
            'filters': self.filters,
            'method': self.method,
            # 'kernel': self.filters,
            # 'bias': self.bias,
            'compression_rate': self.compression_rate,
            'use_bias': self.use_bias,
            'activation': activations.serialize(self.activation),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }
        base_config = super(TransformerShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class OSAR(Layer):

    def __init__(self,
                 units,
                 feature_units,
                 conv_units,
                 num_head=10,
                 attention_dropout=0.0,
                 use_bias=True,
                 gate_initializer='glorot_normal',
                 gate_regularizer='l2',
                 gate_constraint=None,
                 state_initializer='glorot_uniform',
                 state_constraint=None,
                 bias_initializer='zeros',
                 bais_regularizer='l2',
                 bias_constraint=None,
                 **kwargs):
        """OSAR - Object-Stimulated Active Repeater

        # Arguments
            units: int > 0. Number of classic units inside the layer.
            feature_units: int > 0. Number of output features in the layer.
            conv_units: int > 0. Number of convolution units in the layer.
            num_heads: int > 0. Number of attention `heads`.
            attention_dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
            use_bias: Boolean. Whether to use bias (dafault - True).
            gate_initializer: keras.initializer. Initializer for attention weights (default - glorot_normal).
            gate_regularizer: keras.regularizer. Regularizer for attention weights (default - l2).
            gate_constraint: keras.constraint. Constraint for attention weights (default - None).
            state_initializer: keras.initializer. Initializer for state matrices (default - glorot_uniform).
            state_constraint: keras.constraint. Constraint for state matrices (default - None).
            bias_initializer: keras.initializer. Initializer for biases (default - zeros).
            bias_regularizer: keras.regularizer. Regularizer for biases (default - l2).
            bias_constraint: keras.constraint. Constraint for attention weights (default - None).
        
        # Input Shape
            3D tensor with shape: `(batch_size, sequence_length, units)`.
        # Output Shape
            
        # References
            - None yet

        """
        # return_runtime is a flag for testing, which shows the real backend
        # implementation chosen by grappler in graph mode.
        self._return_runtime = kwargs.pop('return_runtime', False)

        super(OSAR, self).__init__(
            dynamic=True,
            **kwargs)

        self.units = units
        self.conv_units = conv_units
        self.feature_units = feature_units
        self.num_head = num_head
        self.attention_dropout = attention_dropout
        self.use_bias = use_bias
        self.state_constraint = constraints.get(state_constraint)
        self.state_initializer = initializers.get(state_initializer)

        self.gate_initializer = initializers.get(gate_initializer)
        self.gate_regularizer = regularizers.get(gate_regularizer)
        self.gate_constraint = constraints.get(gate_constraint)

        self.bias_initializer = initializers.get(bias_initializer)
        self.bais_regularizer = regularizers.get(bais_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        input_shape = input_shape[0]
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `OSAR` layer with non-floating point '
                            'dtype %s' % (dtype,))
        input_shape = tensor_shape.TensorShape(input_shape)
        if tensor_shape.dimension_value(input_shape[-1]) is None:
            raise ValueError('The last dimension of the inputs to `OSAR` '
                             'should be defined. Found `None`.')
        last_dim = tensor_shape.dimension_value(input_shape[-1])
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
        dtr = self.units

        self.p_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='P-Gate',
        )
        self.an_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='An-Gate',
        )
        self.SR_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='R-Gate',
        )
        self.f_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='F-Gate',
        )
        self.d_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='D-Gate',
        )
        self.ao_gate = RelativePartialMultiHeadSelfAttention(
            units=self.units,
            num_head=self.num_head,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            attention_dropout=self.attention_dropout,
            name='R-Gate',
        )

        self.A_state = self.add_weight(
            name='A_state',
            shape=(self.conv_units+self.units, dtr, last_dim,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.O_state = self.add_weight(
            name='O_state',
            shape=(self.conv_units+self.units, dtr, self.feature_units,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)
        self.S_state = self.add_weight(
            name='S_state',
            shape=(last_dim, self.units,),
            constraint=self.state_constraint,
            initializer=self.state_initializer,
            dtype=self.dtype,
            trainable=False)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape[-1]

    # def call(self, inputs, mask=None, training=None, initial_state=None):
        # if not (inputs.shape is 2):
        #     raise ValueError(
        #         'The dimension of the inputs vector should be 2: `(input_shape, reward)`')
        # object_input = inputs[0]
        # reward_input = inputs[1]

        # # Unpacking state matrices
        # object_queries = self.O_state[:, :int(self.O_state.shape[1] / 3), :]
        # object_keys = self.O_state[:, int(
        #     self.O_state.shape[1] / 3):2*int(self.O_state.shape[1] / 3), :]
        # object_values = self.O_state[:, 2*int(
        #     self.O_state.shape[1] / 3):3*int(self.O_state.shape[1] / 3), :]

        # action_queries = self.A_state[:, :int(self.A_state.shape[1] / 3), :]
        # action_keys = self.O_state[:, int(
        #     self.A_state.shape[1] / 3):2*int(self.A_state.shape[1] / 3), :]
        # action_values = self.O_state[:, 2*int(
        #     self.A_state.shape[1] / 3):3*int(self.A_state.shape[1] / 3), :]

        # # Context generator
        # p_object = self.p_gate(
        #     [object_queries, object_keys, object_values, context_bias, relative_bias])
        # object_query_corrected = self.an_gate()

        # Sympathetic circuit

        # Repeater

        # Packing

    def get_config(self):
        config = {
            'units': self.units,
        }
        base_config = super(OSAR, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
