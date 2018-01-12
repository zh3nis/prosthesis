from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
import numpy
from six.moves import xrange
import tensorflow as tf
import c3d_model
import math
import numpy as np
from reader_skipper import MyData


class Config():
  # Training hyperparameters
  is_train = True
  batch_size = 30
  moving_average_decay = 0.9999
  num_epochs = 16
  stable_learning_rate = 1e-4
  finetune_learning_rate = 1e-4
  drop_rate = 0.9
  batch_sampling = 'random'

  # Model hyperparameters
  crop_size = 112
  channels = 1
  num_frames_per_clip = 16
  num_classes = 5
    
  # Other parameters
  model_save_dir = './models'
  model_filename = 'prosthesis_model'
  skipstep=2

  # Training/Validation split
  train_trials = [1, 2, 4, 5]
  # train_trials = [1, 2]
  valid_trials = [3]

    
def placeholder_inputs(config):
  images_placeholder = tf.placeholder(
      tf.float32, shape=(config.batch_size, 
                         config.num_frames_per_clip / config.skipstep,
                         config.crop_size, 
                         config.crop_size,
                         config.channels))
  labels_placeholder = tf.placeholder(tf.int64, shape=(config.batch_size))
  return images_placeholder, labels_placeholder


def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit))
  weight_decay_loss = tf.get_collection('weightdecay_losses')

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + weight_decay_loss 
  return total_loss


def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy


def get_correct_total(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  correct = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
  total = tf.size(correct_pred)
  return correct, total


def _variable_on_cpu(name, shape, initializer):
  #with tf.device('/cpu:0'):
  var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var) * wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var


def run_epoch(session, x, y, data, correct, total, loss, train_op=None):
  corrects = 0
  totals = 0
  losses = 0
  
  for i in range(data.epoch_size):
    images, labels = data.get_batch(i)
    if train_op:
      _, cor, tot, loss_val = session.run(
          [train_op, correct, total, loss], 
          feed_dict={x: images,
                     y: labels[:, -1]})
    else:
      cor, tot, loss_val = session.run(
          [correct, total, loss], 
          feed_dict={x: images,
                     y: labels[:, -1]})
    corrects += cor
    totals += tot
    #print(loss_val.shape)
    losses += np.sum(loss_val)

    if i % (data.epoch_size // 10) == 10 and train_op:
      #saver.save(sess, os.path.join(model_save_dir, 'c3d_ucf_model'), global_step=step)
      print("%.1f train loss: %.3f, acc: %.5f" % (i/data.epoch_size, losses/(i+1), corrects/totals))
    
  return corrects / totals, losses / data.epoch_size


def format_time(duration):
  m, s = divmod(duration, 60)
  h, m = divmod(m, 60)
  return h, m, s


def run():
  global_start_time = time.time()

  config = Config()

  # Create model directory
  if not os.path.exists(config.model_save_dir):
    os.makedirs(config.model_save_dir)
  model_filename = "prosthesis_model"
  
  with tf.Graph().as_default():
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    images_placeholder, labels_placeholder = placeholder_inputs(config)
    tower_grads1 = []
    tower_grads2 = []
    logits = []
    opt_stable = tf.train.AdamOptimizer(config.stable_learning_rate)
    opt_finetuning = tf.train.AdamOptimizer(config.finetune_learning_rate)
    
    with tf.variable_scope('var_name') as var_scope:
      weights = {
          'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 1, 64], 0.0005),
          'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
          'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
          'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], 0.0005),
          'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], 0.0005),
          'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], 0.0005),
          'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], 0.0005),
          'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], 0.0005),
          'wd1': _variable_with_weight_decay('wd1', [8192, 4096], 0.0005),
          'wd2': _variable_with_weight_decay('wd2', [4096, 4096], 0.0005),
          'out': _variable_with_weight_decay('wout', [4096, config.num_classes], 0.0005)
      }
      biases = {
          'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
          'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
          'bc3a': _variable_with_weight_decay('bc3a', [256], 0.000),
          'bc3b': _variable_with_weight_decay('bc3b', [256], 0.000),
          'bc4a': _variable_with_weight_decay('bc4a', [512], 0.000),
          'bc4b': _variable_with_weight_decay('bc4b', [512], 0.000),
          'bc5a': _variable_with_weight_decay('bc5a', [512], 0.000),
          'bc5b': _variable_with_weight_decay('bc5b', [512], 0.000),
          'bd1': _variable_with_weight_decay('bd1', [4096], 0.000),
          'bd2': _variable_with_weight_decay('bd2', [4096], 0.000),
          'out': _variable_with_weight_decay('bout', [config.num_classes], 0.000),
      }

    varlist2 = [ weights['out'],biases['out'] ]
    varlist1 = list( set(weights.values() + biases.values()) - set(varlist2) )
    logit = c3d_model.inference_c3d(
        images_placeholder[ :config.batch_size, : , : , : , : ],
        config.drop_rate,
        config.batch_size,
        weights,
        biases)
    loss_name_scope = ('gpud_0_loss')
    loss = tower_loss(
        loss_name_scope,
        logit,
        labels_placeholder[ :config.batch_size])
    grads1 = opt_stable.compute_gradients(loss, varlist1)
    grads2 = opt_finetuning.compute_gradients(loss, varlist2)
    tower_grads1.append(grads1)
    tower_grads2.append(grads2)
    logits.append(logit)

    logits = tf.concat(logits, 0)
    accuracy = tower_acc(logits, labels_placeholder)
    correct, total = get_correct_total(logits, labels_placeholder)
    grads1 = average_gradients(tower_grads1)
    grads2 = average_gradients(tower_grads2)
    apply_gradient_op1 = opt_stable.apply_gradients(grads1)
    apply_gradient_op2 = opt_finetuning.apply_gradients(grads2, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    train_op = tf.group(apply_gradient_op1, apply_gradient_op2, variables_averages_op)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver(weights.values() + biases.values())
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    
    best_valid_acc = 0.0
    if config.is_train: 
      train_data = MyData(
          path='.', 
          set_indices=config.train_trials, 
          config=config, 
          data_type="training")
    valid_data = MyData(
        path='.', 
        set_indices=config.valid_trials, 
        config=config, 
        data_type="validation")
    
    if config.is_train: 
      for epoch in range(config.num_epochs):
        start_time = time.time()
        
        train_acc, train_loss = run_epoch(
            sess, images_placeholder, labels_placeholder, train_data, correct, total, loss, train_op)
        valid_acc, valid_loss = run_epoch(
            sess, images_placeholder, labels_placeholder, valid_data, correct, total, loss)
        
        duration = time.time() - start_time
        print('Epoch %d train loss: %.3f, acc: %.5f. Valid loss: %.3f, acc: %.5f, duration: %d:%02d:%02d' % \
              ((epoch + 1, train_loss, train_acc, valid_loss, valid_acc) + format_time(duration)))
        
        if valid_acc > best_valid_acc:
          save_path = saver.save(
              sess, os.path.join(config.model_save_dir, config.model_filename))
          print('Valid accuracy improved. Model saved in file: %s' % save_path)
          best_valid_acc = valid_acc
    
    saver.restore(sess, os.path.join(config.model_save_dir, config.model_filename))
    print("Model restored")
    best_valid_acc, best_valid_loss = run_epoch(
        sess, images_placeholder, labels_placeholder, valid_data, correct, total, loss)

  total_duration = time.time() - global_start_time
  print("Done. Best validation loss: %.3f, accuracy: %.5f. Total duration: %d:%02d:%02d" % ((best_valid_loss, best_valid_acc) + format_time(total_duration)))


if __name__ == '__main__':
  run()
