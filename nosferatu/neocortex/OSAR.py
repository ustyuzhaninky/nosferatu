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
import string

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
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import special_math_ops

from nosferatu.neocortex.multi_head_attention import MultiHeadAttention 

_CHR_IDX = string.ascii_lowercase

__all__ = ['OSAR', 'transformerShift', 'MultiHeadAttention']


# def _build_proj_equation(free_dims, bound_dims, output_dims):
#     """Builds an einsum equation for projections inside multi-head attention.
#         From tensorflow-nightly 2.3.x
#         https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/multi_head_attention.py 
#     """
#     input_str = ""
#     kernel_str = ""
#     output_str = ""
#     bias_axes = ""
#     letter_offset = 0
#     for i in range(free_dims):
#         char = _CHR_IDX[i + letter_offset]
#         input_str += char
#         output_str += char

#     letter_offset += free_dims
#     for i in range(bound_dims):
#         char = _CHR_IDX[i + letter_offset]
#         input_str += char
#         kernel_str += char

#     letter_offset += bound_dims
#     for i in range(output_dims):
#         char = _CHR_IDX[i + letter_offset]
#         kernel_str += char
#         output_str += char
#         bias_axes += char
#     equation = "%s,%s->%s" % (input_str, kernel_str, output_str)

#     return equation, bias_axes, len(output_str)


# def _build_attention_equation(rank, attn_axes):
#     """Builds einsum equations for the attention computation.
#     Query, key, value inputs after projection are expected to have the shape as:
#     (bs, <non-attention dims>, <attention dims>, num_heads, channels).
#     bs and <non-attention dims> are treated as <batch dims>.
#     The attention operations can be generalized:
#     (1) Query-key dot product:
#     (<batch dims>, <query attention dims>, num_heads, channels), (<batch dims>,
#     <key attention dims>, num_heads, channels) -> (<batch dims>,
#     num_heads, <query attention dims>, <key attention dims>)
#     (2) Combination:
#     (<batch dims>, num_heads, <query attention dims>, <key attention dims>),
#     (<batch dims>, <value attention dims>, num_heads, channels) -> (<batch dims>,
#     <query attention dims>, num_heads, channels)
#     Args:
#         rank: the rank of query, key, value tensors.
#         attn_axes: a list/tuple of axes, [-1, rank), that will do attention.
#     Returns:
#         Einsum equations.
#     """
#     target_notation = _CHR_IDX[:rank]
#     # `batch_dims` includes the head dim.
#     batch_dims = tuple(np.delete(range(rank), attn_axes + (rank - 1,)))
#     letter_offset = rank
#     source_notation = ""
#     for i in range(rank):
#         if i in batch_dims or i == rank - 1:
#             source_notation += target_notation[i]
#         else:
#             source_notation += _CHR_IDX[letter_offset]
#             letter_offset += 1

#     product_notation = "".join([target_notation[i] for i in batch_dims] +
#                                 [target_notation[i] for i in attn_axes] +
#                                 [source_notation[i] for i in attn_axes])
#     dot_product_equation = "%s,%s->%s" % (source_notation, target_notation,
#                                             product_notation)
#     attn_scores_rank = len(product_notation)
#     combine_equation = "%s,%s->%s" % (product_notation, source_notation,
#                                         target_notation)
#     return dot_product_equation, combine_equation, attn_scores_rank

def transformerShift(inputs, memory, conv_units, short_units,
                     compression_rate:int=2, 
                     method:str='conv',
                     use_bias:bool=True,
                     kernel=None,
                     bias=None,
                     activation=None,
                     ):
    """Compressive Transformer left-shift operation with fixed short and convolution memories
       
       # Arguments
            inputs: A +3D Tensor with shape: `(timesteps, ..., features)`.
            memory: A +3D tensor with shape: `(conv_units+short_units, ..., features)`.
            conv_units: int > 0, Number of convolution memory units in the layer.
            short_units: int > 0, Number of short-term memory units in the layer.
            compression_rate: int >= 1 (default - 2). Compression rate for the selected compression method.
            method: str [`conv`, `mean`, `max`], Defines a memory compression function (default - conv).
                Available compression functions:
                - `conv`: 1D convolution over memory sequence.
                - `mean`: 1D Mean keras.layers.AveragePooling1D.
                - `max`: 1D Max Pooling via keras.MaxPooling1D.

            Only for method=`conv`:
                use_bias: Boolean (default - True), whether the layer uses a bias vector.
                kernel: A 3D Tensor with shape: `(compression_rate, features, features)`.
                bias: A Tensor of shape `(features,)`
                        
            activation: (default - None) Activation functon applied to the output of the compression function ( see keras.activations).

        # Inputs Shape
            +3D tensor with shape: `(batch_size, timesteps, ..., features)`,
            +3D tensor with shape: `(conv_units+short_units, ..., features)`.
        # Output Shape
            +3D tensor with shape: `(batch_size, conv_units+short_units, ..., features)`.
        # References
            - ["Compressive Transformers for Long-Range Sequence Modelling" by Rae et. al.](https://arxiv.org/abs/1911.05507)
    
    """

    if method not in ['conv', 'mean', 'max']:
        raise ValueError(
            f'Compression method `{method}` is not supported.')
    if len(inputs.shape) < 3:
        raise ValueError(
            'The dimension of the input vector'
            ' should be at least 3D: `(batch_size, time_steps, ..., features)`')
    if len(memory.shape) < 2:
        raise ValueError(
            'The dimension of the memory vector'
            ' should be at least 2D: `(conv_units+short_units, ..., features)`')

    if len(inputs.shape) < 3:
        raise ValueError('The dimensions of the first tensor of the inputs to `transformerShift` '
                         f'should be at least 3D: `(batch_size, timesteps, ..., features)`. Found `{inputs.shape}`.')
    if len(memory.shape) < 2:
        raise ValueError('The dimensions of the second tensor of the memory to `transformerShift` '
                         f'should be at least 2D: `(conv_units+short_units, ..., features)`. Found `{memory.shape}`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is None:
        raise ValueError('The last dimension of the first tensor of the inputs to `transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(memory.shape[-1]) is None:
        raise ValueError('The last dimension of the second tensor of the inputs (memory) to `transformerShift` '
                         'should be defined. Found `None`.')
    if tensor_shape.dimension_value(inputs.shape[-1]) is not \
            tensor_shape.dimension_value(memory.shape[-1]):
        raise ValueError('The last dimension of both input tensors to `transformerShift` '
                         f'should match. Found `{tensor_shape.dimension_value(inputs.shape[-1])}` and '
                         f'`{tensor_shape.dimension_value(memory.shape[-1])}`.')
    
    if compression_rate < 1:
        raise ValueError('Compression rate cannot be less than 1.')
    if method in ['max', 'mean']:
        compression_rate = np.int(np.floor(compression_rate))

    activation = activations.get(activation)

    
    timesteps = inputs.shape[1]

    memories = []
    batch_memory = tf.reshape(memory, (1,)+memory.shape)

    for i, batch in enumerate(inputs):
        # Forgetting the oldest
        conv_mem = batch_memory[:, :conv_units]
        short_mem = batch_memory[:, conv_units:]
        old_memory = short_mem[:, :timesteps, ...]

        # Compressing the oldest memories
        if method is 'conv':
            if kernel.shape[1:-1] != batch.shape[1:]:
                raise ValueError(
                    'The dimension of the kernel should be 3D: `(compression_rate, features, features)` '
                    'if shape of the input is 3D: `(batch_size, time_steps, features)`. '
                    f'Found `{kernel.shape[1:-1]}`!=`{batch.shape[1:]}`')
            compression = nn.conv1d(old_memory,
                                    filters=kernel,
                                    stride=compression_rate,
                                    padding='SAME',
                                    name='Compression')
            if use_bias:
                if bias.shape[-1] != compression.shape[-1]:
                    raise ValueError(
                        'The dimension of the bias'
                        ' should be equal to the number of features. '
                        f'Found `{bias.shape[-1]}`!=`{kernel.shape[-1]}`')
                compression = nn.bias_add(
                    compression, bias)
        elif method is 'mean':
            compression = nn.avg_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        elif method is 'max':
            compression = nn.max_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        else:
            compression = nn.avg_pool1d(old_memory, compression_rate,
                                        strides=compression_rate, padding='SAME', name='Compression')
        if compression.shape[-1] > conv_mem.shape[-1]:
            compression = nn.avg_pool1d(compression, compression_rate,
                                        strides=compression_rate, padding='SAME')
        if activation is not None:
            compression = activation(compression)

        # Update Short-Term Memory
        short_mem = K.concatenate([short_mem[:, timesteps:], tf.reshape(batch, (1,)+batch.shape)], axis=1)

        # Update Compressed Memory
        conv_mem = K.concatenate([conv_mem[:, 1:, ...], compression], axis=1)

        batch_memory = K.concatenate([conv_mem, short_mem], axis=1)
        memories.append(batch_memory)

    return K.concatenate(memories, axis=0)


# def multiHeadAttention(query, keys, values, wq, wk, wv, wh):
#     """Multi-head attention operation

#         # Arguments
#             query: A +3D Tensor with shape: `(batch_size, Tq, features_in)`.
#             keys:  A 2D Tensor with shape: `(batch_size, Tv, features_in)`.
#             values: A 2D Tensor with shape: `(batch_size, Tv, features_out)`.
#             wq: A 2D Tensor with shape: `(heads, features_in)`.
#             wk: A 2D Tensor with shape: `(heads, features_in)`.
#             wv: A 2D Tensor with shape: `(heads, features_out)`.
#             wh: A 2D Tensor with shape: `(heads, features_out)`.
       
#         # Output Shape
#             A 3D tensor `(batch_size, Tq, features_out)`.
#         # References
#             - ["Compressive Transformers for Long-Range Sequence Modelling" by Rae et. al.](https://arxiv.org/abs/1911.05507)
#             - ["Neural Turing Machines" by Graves et. al.](http://arxiv.org/abs/1410.5401)
#     """
    
#     if wq.shape[-1] != query.shape[-1]:
#         raise ValueError(
#             'Feature dimension of the `query` and the `wq` should be the the same.'
#             f' Found `{query.shape[-1]}`!={wq.shape[-1]}')
#     if wk.shape[-1] != keys.shape[-1]:
#         raise ValueError(
#             'Feature dimension of the `keys` and the `wk` should be the the same.'
#             f' Found `{keys.shape[-1]}`!={wk.shape[-1]}')
#     if wv.shape[-1] != values.shape[-1]:
#         raise ValueError(
#             'Feature dimension of the `values` and the `wv` should be the the same.'
#             f' Found `{values.shape[-1]}`!={wv.shape[-1]}')
     
#     heads = wq.shape[0]
#     key_dim = wk.shape[-1]
#     free_dims = query_shape.rank - 1

#     equation, bias_axis, op_rank = _build_proj_equation(
#         free_dims, bound_dims=1, output_dims=1
#     )
#     query = tf.einsum(equation, query,)
#     equation, bias_axis, op_rank = _build_proj_equation(
#         keys.shape.rank-1, bound_dims=1, output_dims=1
#     )
#     keys = tf.einsum(equation, keys)
#     equation, bias_axis, op_rank = _build_proj_equation(
#         values.shape.rank-1, bound_dims=1, output_dims=1
#     )   
#     values = tf.einsum(equation, values)

#     query = math_ops.multiply(query, 1.0 / np.sqrt(float(key_dim)))
#     # _attention_axes = tuple(_attention_axes)
#     _dot_product_equation, _combine_equation, attn_scores_rank = (
#         _build_attention_equation(rank, attn_axes=None))
#     attention_scores = special_math_ops.einsum(_dot_product_equation, key,
#                                                query)

#     # att_op = 
#     equation, bias_axis, op_rank = _build_proj_equation(
#         free_dims, bound_dims=2, output_dims=len(output_shape)
#     )
#     advanced_activations.Softmax(axis=norm_axes)
#     output = tf.einsum(equation, att_op)

#     return output


# def contentBasedAttention(query, memory, keys, values, heads, wq, wk, wv):
#     """Multi-head attention operation

#        # Arguments
       
#        # Inputs Shape
#        # Output Shape
#        # References
#             - ["Compressive Transformers for Long-Range Sequence Modelling" by Rae et. al.](https://arxiv.org/abs/1911.05507)
#             - ["Neural Turing Machines" by Graves et. al.](http://arxiv.org/abs/1410.5401)
#     """

#     return None

# class TransformerShift(Layer):
#     """Compressive Transformer left-shift layer with fixed short and convolution memories
       
#        Layer has trainable compression function if method=`conv` is selected.
    
#     """

#     def __init__(self,
#                  short_units,
#                  conv_units,
#                  method='conv',
#                  compression_rate=1,
#                  filters=32,
#                  use_bias=True,
#                  activation=None,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  **kwargs):
#         """Compressive Transformer left-shift layer with fixed short and convolution memories

#         # Arguments
#             feature_units: int > 0, Number of short-term memory units in the layer.
#             conv_units: int > 0, Number of convolution memory units in the layer.
#             method: str [`conv`, `mean`, `max`], Defines a memory compression function (default - conv).
#                 Available compression functions:
#                 - `conv`: 1D convolution over memory sequence.
#                 - `mean`: 1D Mean keras.layers.AveragePooling1D.
#                 - `max`: 1D Max Pooling via keras.MaxPooling1D.
#             compression_rate: int >= 1. Compression rate for the selected compression method.

#             Only for method=`conv`:

#             filters (default - 32): int > 0, Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#             use_bias (default - True): Boolean, whether the layer uses a bias vector.
#             activation (default - None): Activation functon applied to the output of the compression function ( see keras.activations).
#             kernel_initializer (default - 'glorot_uniform'): Initializer for the kernel weights matrix (see keras.initializers).
#             bias_initializer: Initializer for the bias vector (see keras.initializers) (default - 'zeros').
#             kernel_regularizer (default - None): Regularizer function applied to the kernel weights matrix (see keras.regularizers).
#             bias_regularizer (default - None): Regularizer function applied to the bias vector ( see keras.regularizers).
#             kernel_constraint (default - None): Constraint function applied to the kernel matrix ( see keras.constraints).
#             bias_constraint (default - None): Constraint function applied to the bias vector ( see keras.constraints).

#         # Inputs Shape
#             +2D tensor with shape: `(batch_size, ..., features)`,
#             +2D tensor with shape: `(conv_units+short_units, ..., features)`.
#         # Output Shape
#             +3D tensor with shape: `(conv_units+short_units, ..., features)`.
#         # References
#             - [Compressive-Transformer](https://arxiv.org/abs/1911.05507)

#         """

#         # return_runtime is a flag for testing, which shows the real backend
#         # implementation chosen by grappler in graph mode.
#         self._return_runtime = kwargs.pop('return_runtime', False)
#         # Overriding system preset for eager execution at all times
#         kwargs.pop('dynamic', False)
#         super(TransformerShift, self).__init__(
#             dynamic=True,
#             **kwargs)

#         self.conv_units = conv_units
#         self.short_units = short_units
#         self.filters = filters
#         if method not in ['conv', 'mean', 'max']:
#             raise ValueError(
#                 f'Compression method `{method}` is not supported.')
#         self.method = method
#         if compression_rate < 1:
#             raise ValueError('Compression rate cannot be less than 1.')
#         self.compression_rate = compression_rate
#         if method in ['max', 'mean']:
#             self.compression_rate = np.int(np.floor(self.compression_rate))

#         self.use_bias = use_bias
#         self.activation = activations.get(activation)
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)

#     @tf_utils.shape_type_conversion
#     def build(self, input_shape):
#         if len(input_shape) != 2:
#             raise ValueError('The dimensions of the inputs to `TransformerShift` '
#                              f'should be two tensors. Found `{input_shape}`.')
#         if len(input_shape[0]) <= 2:
#             raise ValueError('The dimensions of the first tensor of the inputs to `TransformerShift` '
#                              f'should be at least 3D. Found `{input_shape[0]}`.')
#         if len(input_shape[-1]) <= 2:
#             raise ValueError('The dimensions of the second tensor of the inputs (memory) to `TransformerShift` '
#                              f'should be at least 3D. Found `{input_shape[-1]}`.')
#         dtype = dtypes.as_dtype(self.dtype or K.floatx())
#         vector_shape = tensor_shape.TensorShape(input_shape[0])
#         memory_shape = tensor_shape.TensorShape(input_shape[1])
#         if tensor_shape.dimension_value(vector_shape[-1]) is None:
#             raise ValueError('The last dimension of the first tensor of the inputs to `TransformerShift` '
#                              'should be defined. Found `None`.')
#         if tensor_shape.dimension_value(memory_shape[-1]) is None:
#             raise ValueError('The last dimension of the second tensor of the inputs (memory) to `TransformerShift` '
#                              'should be defined. Found `None`.')
#         if tensor_shape.dimension_value(vector_shape[-1]) is not \
#                 tensor_shape.dimension_value(memory_shape[-1]):
#             raise ValueError('The last dimension of both input tensors to `TransformerShift` '
#                              f'should match. Found `{tensor_shape.dimension_value(vector_shape[-1])}` and '
#                              f'`{tensor_shape.dimension_value(memory_shape[-1])}`.')
#         if self.method is 'conv':
#             self.compression = nn.conv1d
#             self.kernel = self.add_weight(
#                 name='compression_kernel',
#                 shape=(self.compression_rate, vector_shape[-1], self.filters),
#                 initializer=self.kernel_initializer,
#                 regularizer=self.kernel_regularizer,
#                 constraint=self.kernel_constraint,
#                 trainable=True)
#             if self.use_bias:
#                 self.bias = self.add_weight(
#                     name='compression_bias',
#                     shape=(self.filters,),
#                     initializer=self.bias_initializer,
#                     regularizer=self.bias_regularizer,
#                     constraint=self.bias_constraint,
#                     trainable=True)
#         elif self.method is 'mean':
#             self.compression = nn.avg_pool1d
#         elif self.method is 'max':
#             self.compression = nn.max_pool1d
#         else:
#             self.compression = nn.conv1d

#         self.built = True

#     @tf_utils.shape_type_conversion
#     def compute_output_shape(self, input_shape):
#         return input_shape[-1]

#     def call(self, inputs):
#         if len(inputs) != 2:
#             raise ValueError('The dimensions of the inputs to `TransformerShift` '
#                              f'should be two tensors. Found `{inputs.shape}`.')
#         new_tensor_input = inputs[0]
#         memory_input = inputs[1]
#         if len(new_tensor_input.shape) <= 2:
#             raise ValueError(
#                 'The dimension of the input vector'
#                 ' should be at least 3D: `(batch_size, ..., features)`')
#         if len(memory_input.shape) < 3:
#             raise ValueError(
#                 'The dimension of the memory vector'
#                 ' should be at least 3D: `(conv_units+short_units, ..., features)`')
#         conv_mem = memory_input[:self.conv_units]
#         short_mem = memory_input[self.conv_units:]
#         d_mask = new_tensor_input.shape[0]
#         # Forgetting the oldest
#         old_mem = tf.reshape(
#             short_mem[:d_mask, ...], (d_mask,)+short_mem.shape[1:])
#         # Compressing the oldest memories

#         if self.method is 'conv':
#             compression = nn.conv1d(old_mem,
#                                     filters=self.kernel,
#                                     stride=self.compression_rate,
#                                     padding='SAME',
#                                     name='Compression')
#             if self.use_bias:
#                 compression = nn.bias_add(
#                     compression, self.bias)
#         elif self.method is 'mean':
#             compression = nn.avg_pool1d(old_mem, self.compression_rate,
#                                         strides=self.compression_rate, padding='SAME', name='Compression')
#         elif self.method is 'max':
#             compression = nn.max_pool1d(old_mem, self.compression_rate,
#                                         strides=self.compression_rate, padding='SAME', name='Compression')
#         else:
#             compression = nn.avg_pool1d(old_mem, self.compression_rate,
#                                         strides=self.compression_rate, padding='SAME', name='Compression')
#         if compression.shape[-1] > 1:
#             compression = nn.avg_pool1d(compression, 1,
#                                         strides=1, padding='SAME')
#         if self.activation is not None:
#             compression = self.activation(compression)

#         new_convs = tf.reshape(
#             compression, (d_mask,)+short_mem.shape[1:])

#         # Update Short-Term Memory
#         short_mem = K.concatenate(
#             [short_mem[:-d_mask, ...], new_tensor_input], axis=0)
#         # Update Compressed Memory
#         conv_mem = K.concatenate(
#             [conv_mem, new_convs], axis=0)[d_mask:, ...]
#         # conv_mem = tf.nn.convolution(conv_mem, compression)

#         return K.concatenate([conv_mem, short_mem], axis=0)

#     def get_config(self):
#         config = {
#             'conv_units': self.conv_units,
#             'short_units': self.short_units,
#             'filters': self.filters,
#             'method': self.method,
#             # 'kernel': self.filters,
#             # 'bias': self.bias,
#             'compression_rate': self.compression_rate,
#             'use_bias': self.use_bias,
#             'activation': activations.serialize(self.activation),
#             'kernel_initializer': initializers.serialize(self.kernel_initializer),
#             'bias_initializer': initializers.serialize(self.bias_initializer),
#             'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
#             'bias_regularizer': regularizers.serialize(self.bias_regularizer),
#             'kernel_constraint': constraints.serialize(self.kernel_constraint),
#             'bias_constraint': constraints.serialize(self.bias_constraint),
#         }
#         base_config = super(TransformerShift, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

class OSAR(Layer):

    def __init__(self,
                 units,
                 feature_units,
                 conv_units,
                 n_actions,
                 num_head=10,
                 dropout=0.0,
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
            n_actions: int > 1. Number of actions in the output vector.
            num_heads: int > 0. Number of attention `heads`.
            dropout: 0.0 <= float <= 1.0. Dropout rate inside attention weights.
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
        self.n_actions = n_actions
        self.conv_units = conv_units
        self.feature_units = feature_units
        self.num_head = num_head
        self.dropout = dropout
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
        n_digits = tensor_shape.dimension_value(input_shape[-1])
        batch_dim = tensor_shape.dimension_value(input_shape[0])
        self.input_spec = InputSpec(min_ndim=2, axes={-1: n_digits})
        dtr = self.units

        # Probability gate
        self.p_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=n_digits,
            value_dim=n_digits,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='P-Gate',
        )

        # New action gate
        self.an_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=n_digits,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='An-Gate',
        )

        # Stimuli-Reward gate
        self.SR_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=self.n_actions,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='SR-Gate',
        )
        
        # Forecast gate
        self.f_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=self.n_actions,
            value_dim=n_digits,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='F-Gate',
        )

        # Backprop decision gate
        self.d_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=n_digits,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='D-Gate',
        )

        # Action overhaul gate
        self.ao_gate = MultiHeadAttention(
            # units=self.units,
            num_head=self.num_head,
            key_dim=self.n_actions,
            value_dim=self.n_actions,
            use_bias=self.use_bias,
            kernel_initializer=self.gate_initializer,
            kernel_regularizer=self.gate_regularizer,
            kernel_constraint=self.gate_constraint,
            bias_initializer=self.bias_initializer,
            bias_regularizer=self.bais_regularizer,
            bias_constraint=self.bias_constraint,
            dropout=self.dropout,
            name='AO-Gate',
        )

        self.A_state = self.add_weight(
            name='A_state',
            shape=(self.conv_units+self.units, dtr, n_digits,),
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
            shape=(n_digits, self.units,),
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
