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
"""Compact implementation of a simplified Rainbow agent.

Specifically, we implement the following components from Rainbow:

  * n-step updates;
  * prioritized replay; and
  * distributional RL.

These three components were found to significantly impact the performance of
the Atari game-playing agent.

Furthermore, our implementation does away with some minor hyperparameter
choices. Specifically, we

  * keep the beta exponent fixed at beta=0.5, rather than increase it linearly;
  * remove the alpha parameter, which was set to alpha=0.5 throughout the paper.

Details in "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel et al. (2018).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nosferatu.dopamine.agents.dqn import dqn_agent
from nosferatu.dopamine.agents.rainbow import rainbow_agent
from nosferatu.dopamine.discrete_domains import atari_lib
from nosferatu import hunter_lib
from nosferatu.dopamine.replay_memory import prioritized_replay_buffer
import tensorflow as tf

import gin.tf

@gin.configurable
class NosferatuAgent(rainbow_agent.RainbowAgent):
  def __init__(self,
               sess,
               num_actions,
               num_capsule,
               dim_capsule,
               routings=3,
               observation_shape=(84, 84),
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=4,
               network=hunter_lib.RainbowCapsuleHunter,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=5,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/gpu:0',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    """Initializes the agent and constructs the components of its graph.

    Args:
      sess: `tf.Session`, for executing ops.
      num_actions: int, number of actions the agent can take at any state.
      observation_shape: tuple of ints or an int. If single int, the observation
        is assumed to be a 2D square.
      observation_dtype: tf.DType, specifies the type of the observations. Note
        that if your inputs are continuous, you should set this to tf.float32.
      stack_size: int, number of frames to use in state stack.
      network: tf.Keras.Model, expects four parameters:
        (num_actions, num_atoms, support, network_type).  This class is used to
        generate network instances that are used by the agent. Each
        instantiation would have different set of variables. See
        dopamine.discrete_domains.atari_lib.RainbowNetwork as an example.
      num_atoms: int, the number of buckets of the value function distribution.
      vmax: float, the value distribution support is [-vmax, vmax].
      gamma: float, discount factor with the usual RL meaning.
      update_horizon: int, horizon at which updates are performed, the 'n' in
        n-step update.
      min_replay_history: int, number of transitions that should be experienced
        before the agent begins training its value function.
      update_period: int, period between DQN updates.
      target_update_period: int, update period for the target network.
      epsilon_fn: function expecting 4 parameters:
        (decay_period, step, warmup_steps, epsilon). This function should return
        the epsilon value used for exploration during training.
      epsilon_train: float, the value to which the agent's epsilon is eventually
        decayed during training.
      epsilon_eval: float, epsilon used when evaluating the agent.
      epsilon_decay_period: int, length of the epsilon decay schedule.
      replay_scheme: str, 'prioritized' or 'uniform', the sampling scheme of the
        replay memory.
      tf_device: str, Tensorflow device on which the agent's graph is executed.
      use_staging: bool, when True use a staging area to prefetch the next
        training batch, speeding training up by about 30%.
      optimizer: `tf.train.Optimizer`, for training the value function.
      summary_writer: SummaryWriter object for outputting training statistics.
        Summary writing disabled if set to None.
      summary_writing_frequency: int, frequency with which summaries will be
        written. Lower values will result in slower training.
    """  
    
    self.optimizer = optimizer
    self._num_capsule = num_capsule
    self._dim_capsule = dim_capsule
    self._routings = routings

    rainbow_agent.RainbowAgent.__init__(
        self,
        sess=sess,
        num_actions=num_actions,
        num_atoms=num_atoms,
        replay_scheme=replay_scheme,
        observation_shape=observation_shape,
        observation_dtype=observation_dtype,
        stack_size=stack_size,
        network=network,
        gamma=gamma,
        vmax=vmax,
        update_horizon=update_horizon,
        min_replay_history=min_replay_history,
        update_period=update_period,
        target_update_period=target_update_period,
        epsilon_fn=epsilon_fn,
        epsilon_train=epsilon_train,
        epsilon_eval=epsilon_eval,
        epsilon_decay_period=epsilon_decay_period,
        tf_device=tf_device,
        use_staging=use_staging,
        optimizer=self.optimizer,
        summary_writer=summary_writer,
        summary_writing_frequency=summary_writing_frequency)
    
  def _create_network(self, name):
    """Builds a convolutional network that outputs Q-value distributions.

    Args:
      name: str, this name is passed to the tf.keras.Model and used to create
        variable scope under the hood by the tf.keras.Model.
    Returns:
      network: tf.keras.Model, the network instantiated by the Keras model.
    """
    network = self.network(self.num_actions, self._num_atoms, self._support,
                           self._num_capsule, self._dim_capsule, routings=self._routings,
                           name=name)
    return network
  
  def _select_action(self):
    """Select an action from the set of available actions.

    Chooses an action randomly with probability self._calculate_epsilon(), and
    otherwise acts greedily according to the current Q-value estimates.

    Returns:
       int, the selected action.
    """
    if self.eval_mode:
      epsilon = self.epsilon_eval
    else:
      epsilon = self.epsilon_fn(
          self.epsilon_decay_period,
          self.training_steps,
          self.min_replay_history,
          self.epsilon_train)
    
    # take a winning action
    # if not tf.executing_eagerly():
    #   return self._sess.run(self._q_argmax, {self.state_ph: self.state})
    # else:
    #   self._net_outputs = self.online_convnet(self.state)
    #   return tf.argmax(self._net_outputs.q_values, axis=1)[0]
    
    # Take a bold action (high probability) or a random one
    if random.random() <= epsilon:
      # Choose a random action with probability epsilon.
      return random.randint(0, self.num_actions - 1)
    else:
      # Choose the action with highest Q-value at the current state.
      if not tf.executing_eagerly():
        return self._sess.run(self._q_argmax, {self.state_ph: self.state})
      else:
        self._net_outputs = self.online_convnet(self.state)
        return tf.argmax(self._net_outputs.q_values, axis=1)[0]
