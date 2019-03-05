# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for training."""

import six

import tensorflow as tf
from deeplab.core import preprocess_utils

slim = tf.contrib.slim


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.

  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    scope: String, the scope for the loss.

  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    # not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,ignore_label)) * loss_weight

    loss_weight0 = 1.5
    loss_weight1 = 2.3
    loss_weight2 = 2.5
    loss_weight3 = 2
    loss_weight4 = 2
    loss_weight5 = 2
    loss_weight6 = 2
    loss_weight7 = 2
    loss_weight8 = 4.5
    loss_weight9 = 0
    loss_weight10 = 2
    loss_weight11 = 0
    loss_weight12 = 1.5
    loss_weight13 = 0
    loss_weight14 = 2
    loss_weight15 = 2
    loss_weight16 = 2
    loss_weight17 = 0
    loss_weight18 = 2.5
    loss_weight19 = 5
    loss_weight20 = 5
    loss_weight21 = 10
    loss_weight22 = 5
    loss_weight23 = 0
    loss_weight24 = 3
    loss_weight25 = 10
    loss_weight26 = 10
    loss_weight27 = 10
    loss_weight28 = 0
    loss_weight29 = 0
    loss_weight30 = 0
    loss_weight31 = 20
    loss_weight32 = 4
    loss_weight33 = 12
    loss_weight34 = 4
    loss_weight35 = 4
    loss_weight_ignore = 0

    not_ignore_mask =   tf.to_float(tf.equal(scaled_labels, 0)) * loss_weight0 + \
                        tf.to_float(tf.equal(scaled_labels, 1)) * loss_weight1 + \
                        tf.to_float(tf.equal(scaled_labels, 2)) * loss_weight2 + \
                        tf.to_float(tf.equal(scaled_labels, 3)) * loss_weight3 + \
                        tf.to_float(tf.equal(scaled_labels, 4)) * loss_weight4 + \
                        tf.to_float(tf.equal(scaled_labels, 5)) * loss_weight5 + \
                        tf.to_float(tf.equal(scaled_labels, 6)) * loss_weight6 + \
                        tf.to_float(tf.equal(scaled_labels, 7)) * loss_weight7 + \
                        tf.to_float(tf.equal(scaled_labels, 8)) * loss_weight8 + \
                        tf.to_float(tf.equal(scaled_labels, 9)) * loss_weight9 + \
                        tf.to_float(tf.equal(scaled_labels, 10)) * loss_weight10 + \
                        tf.to_float(tf.equal(scaled_labels, 11)) * loss_weight11 + \
                        tf.to_float(tf.equal(scaled_labels, 12)) * loss_weight12 + \
                        tf.to_float(tf.equal(scaled_labels, 13)) * loss_weight13 + \
                        tf.to_float(tf.equal(scaled_labels, 14)) * loss_weight14 + \
                        tf.to_float(tf.equal(scaled_labels, 15)) * loss_weight15 + \
                        tf.to_float(tf.equal(scaled_labels, 16)) * loss_weight16 + \
                        tf.to_float(tf.equal(scaled_labels, 17)) * loss_weight17 + \
                        tf.to_float(tf.equal(scaled_labels, 18)) * loss_weight18 + \
                        tf.to_float(tf.equal(scaled_labels, 19)) * loss_weight19 + \
                        tf.to_float(tf.equal(scaled_labels, 20)) * loss_weight20 + \
                        tf.to_float(tf.equal(scaled_labels, 21)) * loss_weight21 + \
                        tf.to_float(tf.equal(scaled_labels, 22)) * loss_weight22 + \
                        tf.to_float(tf.equal(scaled_labels, 23)) * loss_weight23 + \
                        tf.to_float(tf.equal(scaled_labels, 24)) * loss_weight24 + \
                        tf.to_float(tf.equal(scaled_labels, 25)) * loss_weight25 + \
                        tf.to_float(tf.equal(scaled_labels, 26)) * loss_weight26 + \
                        tf.to_float(tf.equal(scaled_labels, 27)) * loss_weight27 + \
                        tf.to_float(tf.equal(scaled_labels, 28)) * loss_weight28 + \
                        tf.to_float(tf.equal(scaled_labels, 29)) * loss_weight29 + \
                        tf.to_float(tf.equal(scaled_labels, 30)) * loss_weight30 + \
                        tf.to_float(tf.equal(scaled_labels, 31)) * loss_weight31 + \
                        tf.to_float(tf.equal(scaled_labels, 32)) * loss_weight32 + \
                        tf.to_float(tf.equal(scaled_labels, 33)) * loss_weight33 + \
                        tf.to_float(tf.equal(scaled_labels, 34)) * loss_weight34 + \
                        tf.to_float(tf.equal(scaled_labels, 35)) * loss_weight35 + \
                        tf.to_float(tf.equal(scaled_labels, ignore_label)) * loss_weight_ignore

    one_hot_labels = slim.one_hot_encoding(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)
    tf.losses.softmax_cross_entropy(
        one_hot_labels,
        tf.reshape(logits, shape=[-1, num_classes]),
        weights=not_ignore_mask,
        scope=loss_scope)


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.

  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.

  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step','logits']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = slim.get_variables_to_restore(exclude=exclude_list)

  if variables_to_restore:
    return slim.assign_from_checkpoint_fn(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.

  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.

  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.

  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in slim.get_model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(
    learning_policy, base_learning_rate, learning_rate_decay_step,
    learning_rate_decay_factor, training_number_of_steps, learning_power,
    slow_start_step, slow_start_learning_rate):
  """Gets model's learning rate.

  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power

  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.

  Returns:
    Learning rate for the specified learning policy.

  Raises:
    ValueError: If learning policy is not recognized.
  """
  global_step = tf.train.get_or_create_global_step()
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        global_step,
        training_number_of_steps,
        end_learning_rate=0,
        power=learning_power)
  else:
    raise ValueError('Unknown learning policy.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step, slow_start_learning_rate,
                  learning_rate)
