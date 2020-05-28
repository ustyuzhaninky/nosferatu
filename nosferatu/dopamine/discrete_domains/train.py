# coding=utf-8
# Copyright 2018 The Dopamine Authors.
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
# limitations under the Lzicense.
r"""The entry point for running a Dopamine agent.

python -um nosferatu.dopamine.discrete_domains.train --base_dir=..\tmp\rainbow --gin_files=D:\Projects\python\PhDSub\TowerProject\nosferatu\nosferatu\dopamine\agents\rainbow\configs\rainbow_otc.gin

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



from absl import app
from absl import flags

from nosferatu.dopamine.discrete_domains import run_experiment

import tensorflow as tf
from absl import logging
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

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
  logging.set_verbosity(logging.INFO)
  run_experiment.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
  runner = run_experiment.create_runner(FLAGS.base_dir)
  runner.run_experiment()


if __name__ == '__main__':
  flags.mark_flag_as_required('base_dir')
  app.run(main)
