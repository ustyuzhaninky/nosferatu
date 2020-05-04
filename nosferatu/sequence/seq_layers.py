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
from tensorflow.python.util import nest
from tensorflow.python.framework import auto_control_deps
from tensorflow.compat.v1.keras.preprocessing.sequence import pad_sequences
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
        #   self._buffer = self.add_weight(
        #       name='buffer',
        #       shape=(self.n_steps,) + input_shape[1:], 
        #       initializer='zeros', trainable=False)
          self.built = True
      
    def call(self, inputs, buffer):
        input_shape = inputs.shape
        buffer = buffer
        
        if input_shape[0] > self.n_steps:
            raise ValueError(
                    'The number of steps in the buffer cannot be smaller than the batch dimension.'
                )

        masked_inputs =  tf.keras.layers.concatenate([inputs, tf.zeros((self.n_steps-input_shape[0],)+input_shape[1:])], axis=0)
        nullifying_mask = tf.keras.layers.concatenate([tf.zeros(input_shape), tf.ones((self.n_steps-input_shape[0],)+input_shape[1:])], axis=0)
        
        rolled = tf.roll(buffer, shift=input_shape[0], axis=0)
        updated_buffer = rolled * nullifying_mask + masked_inputs

        output = tf.reshape(updated_buffer, 
                            (inputs.shape[0], 
                            int(self.n_steps / inputs.shape[0]),) + updated_buffer.shape[1:])
        return output, updated_buffer
    
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
        super(Buffer, self).__init__(trainable=False, dynamic=True, **kwargs)
        # if max_shift and max_shift >= n_steps:
        #     raise ValueError('Maximum shift must smaller than number of steps in the buffer: `max_shift > n_steps`.')

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
    
    def __call__(self, inputs, *args, **kwargs):
      """Wraps `call`, applying pre- and post-processing steps.
      Arguments:
        inputs: input tensor(s).
        *args: additional positional arguments to be passed to `self.call`.
        **kwargs: additional keyword arguments to be passed to `self.call`.
      Returns:
        Output tensor(s).
      Note:
        - The following optional keyword arguments are reserved for specific uses:
          * `training`: Boolean scalar tensor of Python boolean indicating
            whether the `call` is meant for training or inference.
          * `mask`: Boolean input mask.
        - If the layer's `call` method takes a `mask` argument (as some Keras
          layers do), its default value will be set to the mask generated
          for `inputs` by the previous layer (if `input` did come from
          a layer that generated a corresponding mask, i.e. if it came from
          a Keras layer with masking support.
      Raises:
        ValueError: if the layer's `call` method returns None (an invalid value).
      """
      call_context = base_layer_utils.call_context()
      input_list = nest.flatten(inputs)

      if self.cell.buffer is None:
        self.cell.buffer = self.cell.add_weight(
            name='buffer',
            shape=(self.n_steps,) + inputs.shape[1:], 
            initializer='zeros', trainable=False)
      buffer = self.cell.buffer

      # We will attempt to build a TF graph if & only if all inputs are symbolic.
      # This is always the case in graph mode. It can also be the case in eager
      # mode when all inputs can be traced back to `keras.Input()` (when building
      # models using the functional API).
      build_graph = tf_utils.are_all_symbolic_tensors(input_list)

      # Accept NumPy and scalar inputs by converting to Tensors.
      if any(isinstance(x, (np.ndarray, float, int)) for x in input_list):
        def _convert_non_tensor(x):
          # Don't call `ops.convert_to_tensor` on all `inputs` because
          # `SparseTensors` can't be converted to `Tensor`.
          if isinstance(x, (np.ndarray, float, int)):
            return ops.convert_to_tensor(x)
          return x
        inputs = nest.map_structure(_convert_non_tensor, inputs)
        input_list = nest.flatten(inputs)

      # Handle `mask` propagation from previous layer to current layer. Masks can
      # be propagated explicitly via the `mask` argument, or implicitly via
      # setting the `_keras_mask` attribute on the inputs to a Layer. Masks passed
      # explicitly take priority.
      mask_arg_passed_by_framework = False
      input_masks = self._collect_input_masks(inputs, args, kwargs)
      if (self._expects_mask_arg and input_masks is not None and
          not self._call_arg_was_passed('mask', args, kwargs)):
        mask_arg_passed_by_framework = True
        kwargs['mask'] = input_masks

      # If `training` argument was not explicitly passed, propagate `training`
      # value from this layer's calling layer.
      training_arg_passed_by_framework = False
      # Priority 1: `training` was explicitly passed.
      if self._call_arg_was_passed('training', args, kwargs):
        training_value = self._get_call_arg_value('training', args, kwargs)
        if not self._expects_training_arg:
          kwargs.pop('training')
      else:
        training_value = None
        # Priority 2: `training` was passed to a parent layer.
        if call_context.training is not None:
          training_value = call_context.training
        # Priority 3a: `learning_phase()` has been set.
        elif K.global_learning_phase_is_set():
          training_value = K.learning_phase()
        # Priority 3b: Pass the `learning_phase()` if in the Keras FuncGraph.
        elif build_graph:
          with K.get_graph().as_default():
            if base_layer_utils.is_in_keras_graph():
              training_value = K.learning_phase()

        if self._expects_training_arg and training_value is not None:
          # Force the training_value to be bool type which matches to the contract
          # for layer/model call args.
          if tensor_util.is_tensor(training_value):
            training_value = math_ops.cast(training_value, dtypes.bool)
          else:
            training_value = bool(training_value)
          kwargs['training'] = training_value
          training_arg_passed_by_framework = True

      # Only create Keras history if at least one tensor originates from a
      # `keras.Input`. Otherwise this Layer may be being used outside the Keras
      # framework.
      if build_graph and base_layer_utils.needs_keras_history(inputs):
        base_layer_utils.create_keras_history(inputs)

      # Clear eager losses on top level model call.
      # We are clearing the losses only on the top level model call and not on
      # every layer/model call because layer/model may be reused.
      if (base_layer_utils.is_in_eager_or_tf_function() and
          not call_context.in_call):
        self._clear_losses()

      with call_context.enter(self, inputs, build_graph, training_value):
        # Check input assumptions set after layer building, e.g. input shape.
        if build_graph:
          input_spec.assert_input_compatibility(self.input_spec, inputs,
                                                self.name)
          if (any(isinstance(x, ragged_tensor.RaggedTensor) for x in input_list)
              and self._supports_ragged_inputs is False):  # pylint: disable=g-bool-id-comparison
            raise ValueError('Layer %s does not support RaggedTensors as input. '
                            'Inputs received: %s. You can try converting your '
                            'input to an uniform tensor.' % (self.name, inputs))

          graph = K.get_graph()
          with graph.as_default(), K.name_scope(self._name_scope()):
            # Build layer if applicable (if the `build` method has been
            # overridden).
            self._maybe_build(inputs)
            cast_inputs = self._maybe_cast_inputs(inputs)

            # Wrapping `call` function in autograph to allow for dynamic control
            # flow and control dependencies in call. We are limiting this to
            # subclassed layers as autograph is strictly needed only for
            # subclassed layers and models.
            # tf_convert will respect the value of autograph setting in the
            # enclosing tf.function, if any.
            if (base_layer_utils.is_subclassed(self) and
                not base_layer_utils.from_saved_model(self)):
              call_fn = autograph.tf_convert(
                  self.call, ag_ctx.control_status_ctx())
            else:
              call_fn = self.call

            if not self.dynamic:
              try:
                with base_layer_utils.autocast_context_manager(
                    self._compute_dtype):
                  # Add auto_control_deps in V2 when they are not already added by
                  # a `tf.function`.
                  if (ops.executing_eagerly_outside_functions() and
                      not base_layer_utils.is_in_eager_or_tf_function()):
                    with auto_control_deps.AutomaticControlDependencies() as acd:
                      outputs, buffer = call_fn(cast_inputs, buffer, *args, **kwargs)
                      # Wrap Tensors in `outputs` in `tf.identity` to avoid
                      # circular dependencies.
                      outputs = base_layer_utils.mark_as_return(outputs, acd)
                      buffer = base_layer_utils.mark_as_return(buffer, acd)
                  else:
                    outputs, buffer = call_fn(cast_inputs, buffer, *args, **kwargs)

              except errors.OperatorNotAllowedInGraphError as e:
                raise TypeError('You are attempting to use Python control '
                                'flow in a layer that was not declared to be '
                                'dynamic. Pass `dynamic=True` to the class '
                                'constructor.\nEncountered error:\n"""\n' +
                                str(e) + '\n"""')
            else:
              outputs = self._symbolic_call(inputs)

            if outputs is None:
              raise ValueError('A layer\'s `call` method should return a '
                              'Tensor or a list of Tensors, not None '
                              '(layer: ' + self.name + ').')
            if base_layer_utils.have_all_keras_metadata(inputs):
              if training_arg_passed_by_framework:
                kwargs.pop('training')
              if mask_arg_passed_by_framework:
                kwargs.pop('mask')
              inputs, outputs = self._set_connectivity_metadata_(
                  inputs, outputs, args, kwargs)
            self._handle_activity_regularization(inputs, outputs)
            self._set_mask_metadata(inputs, outputs, input_masks)
            if hasattr(self, '_set_inputs') and not self.inputs:
              self._set_inputs(inputs, outputs)
        else:
          # Eager execution on data tensors.
          with K.name_scope(self._name_scope()):
            self._maybe_build(inputs)
            cast_inputs = self._maybe_cast_inputs(inputs)
            with base_layer_utils.autocast_context_manager(
                self._compute_dtype):
              outputs, buffer = self.call(cast_inputs, buffer, *args, **kwargs)
              
            self._handle_activity_regularization(inputs, outputs)
            self._set_mask_metadata(inputs, outputs, input_masks)

      updates = []
      updates.append(state_ops.assign(self.cell.buffer, buffer))
        
      self.add_update(updates)

      return outputs

    def call(self, inputs, buffer):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        inputs, row_lengths = K.convert_inputs_if_ragged(inputs)

        def step(inputs, buffer):
            output, new_buffer = self.cell.call(
                inputs, buffer)
            return output, new_buffer

        output, buffer = step(inputs, buffer)

        print(output.shape, buffer.shape, self.cell.buffer.shape)
        # updates = []
        # updates.append(state_ops.assign(self.cell.buffer, buffer))
        
        # self.add_update(updates)

        return output, buffer
    
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

#python -um nosferatu.sequence.seq_layers
# from tensorflow.python.framework.ops import disable_eager_execution
# # disable_eager_execution()
# inputs = tf.keras.Input(shape=(2), batch_size=1)
# outputs = Buffer(5)(inputs)
# model = tf.keras.Model(inputs, outputs)

# model.compile(optimizer='adam', loss='mse')
# model.summary()
# x = np.random.randn(100, 2)

# # for i in range(3):
# #     model.predict(x[i:(2*i+1)])
# print('AAAA', f' x={x[0].reshape(1, -1)}', f' y={model.predict(x[0].reshape(1, -1))}')
# print('BBBB', f' x={x[1:3].reshape(2, -1)}', f' y={model.predict(x[1:3].reshape(2, -1))}')
