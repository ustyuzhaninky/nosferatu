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
r"""The entry point for running a Dopamine agent.

Example:
    python -um nosferatu.train --base_dir=..\tmp\nosferatu_unet --gin_files=D:\Jorgen\Projects\python\PhDSub\TowerProject\nosferatu\nosferatu\configs\nosferatu_otc.gin
    python -um nosferatu.train --base_dir=\tmp\otctest --gin_files=nosferatu_otc.gin
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

from nosferatu import run_experiment

import tensorflow as tf
from absl import logging
from tensorflow.python.framework.ops import disable_eager_execution

# disable_eager_execution()
# tf.enable_v2_behavior()
tf.config.experimental_run_functions_eagerly(True)
# logging.set_verbosity(logging.ERROR)
# tf.disable_tensor_equality()

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def main(unused_argv):
  """Main method.

  Args:
    unused_argv: Arguments (unused).
  """
  print(FLAGS.gin_files)
  # logging.set_verbosity(logging.INFO)
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = run_experiment.create_runner(FLAGS.base_dir)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)

