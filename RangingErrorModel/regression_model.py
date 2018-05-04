from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import math
import argparse
import tensorflow as tf
# data import library

sys.path.append("../tools/")
import data_tools

TRAIN_DIR = './'
BATCH_SIZE = 200
MAX_STEPS = 20000


def inference_graph(input, input_size, keep_prob):
    # NN structure parameters
    L1_patch_w = 4
    L1_patch_h = 1
    L1_depth = 10
    L2_patch_w = 5
    L2_patch_h = 1
    L2_depth = 20
    L3_patch_w = 4
    L3_patch_h = 1
    L3_depth = 20
    L4_patch_w = 4
    L4_patch_h = 1
    L4_depth = 40
    fc_size = 128
    fc_input_size = (input_size - L1_patch_w - L2_patch_w - 4 * L3_patch_w - 4 * L4_patch_w + 25) // 16
    
    x_input = tf.reshape(input, [-1,input_size,1,1])
    
    # Convolutional 1
    with tf.name_scope('conv1'):
        weights = tf.Variable(tf.truncated_normal([L1_patch_w,L1_patch_h,1,L1_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L1_depth]), name='biases')
        conv1 = tf.nn.relu(tf.nn.conv2d(x_input, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)
    # Convolutional 2
    with tf.name_scope('conv2'):
        weights = tf.Variable(tf.truncated_normal([L2_patch_w,L2_patch_h,L1_depth,L2_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L2_depth]), name='biases')
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, weights, strides=[1, 2, 1, 1], padding='SAME') + biases)
    # Pooling 1
    with tf.name_scope('pool1'):
        pool1 = tf.nn.max_pool(conv2, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
    # Convolutional 3
    with tf.name_scope('conv3'):
        weights = tf.Variable(tf.truncated_normal([L3_patch_w,L3_patch_h,L2_depth,L3_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L3_depth]), name='biases')
        conv3 = tf.nn.relu(tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='VALID') + biases)
    # Convolutional 4
    with tf.name_scope('conv4'):
        weights = tf.Variable(tf.truncated_normal([L4_patch_w,L4_patch_h,L3_depth,L4_depth], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[L4_depth]), name='biases')
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, weights, strides=[1, 2, 1, 1], padding='SAME') + biases)
    # Pool 2
    with tf.name_scope('pool2'):
        pool2 = tf.nn.max_pool(conv4, ksize=[1,2,1,1], strides=[1,2,1,1], padding='VALID')
    # Fully connected layer
    with tf.name_scope('fc'):
        weights = tf.Variable(tf.truncated_normal([fc_input_size * L4_depth, fc_size], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[fc_size]), name='biases')
        pool2_flat = tf.reshape(pool2, [-1, fc_input_size * L4_depth])
        fc = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)
    # Dropout layer
    with tf.name_scope('dropout'):
        #keep_prob = tf.placeholder(tf.float32)
        fc_drop = tf.nn.dropout(fc, keep_prob)
    # Readout Layer
    with tf.name_scope('readout'):
        weights =  tf.Variable(tf.truncated_normal([fc_size, 1], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.1, shape=[1]), name='biases')
        regression = tf.matmul(fc_drop, weights) + biases
        
    # save inference graph to file
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "regression.pbtxt", as_text=True)
        
    return regression
    
def training_graph(logits_, labels_, learning_rate):
    """Build the training graph.
    
    Args:
        logits: Logits tensor, float - [BATCH_SIZE, NUM_CLASSES].
        labels: Labels tensor, int32 - [BATCH_SIZE, NUM_CLASSES].
        learning_rate: The learning rate for selected optimizer
    Returns:
        train_op: The Op for training.
        loss: The Op for calculating loss.
    """
    # Create an operation that calculates loss.
    labels_ = tf.to_double(labels_)
    ms_error = tf.losses.mean_squared_error(labels=labels_, predictions=logits_)  #loss_collection=tf.GraphKeys.LOSSES) 
    loss = tf.reduce_sum(ms_error, name='mean_squared_error_sum')
    # Create ADAM optimizer with given learning rate.
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step.
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss and increment the global step counter
    train_op = optimizer.minimize(loss, global_step=global_step)
    # save train graph to file
    tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "train_regression.pbtxt", as_text=True)
    
    return train_op, loss
    
FLAGS = None

def main(_):
    # import data
    start_time = time.time()
    
    data = data_tools.error_regression_dataset(['../data/evaluation/dataset1/tag_room0.csv', '../data/evaluation/dataset2/tag_room0.csv'], split_factor=0.6, scaling=True)
    print("rows: %d, columns: %d" % (data.train.samples.shape[0], data.train.samples.shape[1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    
    # prepare folder to output model
    try:
        os.makedirs(TRAIN_DIR)
    except OSError:
        if os.path.exists(TRAIN_DIR):
            # we are nearly safe
            pass
        else:
            # there was an error on creation
            raise
    
    checkpoint_file = os.path.join(TRAIN_DIR, 'checkpoint_regression')
       
    # Build the complete graph for feeding inputs, training, and saving checkpoints
    graph = tf.Graph()
    with graph.as_default():
        # Generate placeholders for input samples and labels.
        x = tf.placeholder(tf.float32, [None, data.train.samples.shape[1]], name='input_data')
        labels = tf.placeholder(tf.float32, [None, 1], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        tf.add_to_collection("inputs", x)       # Remember this Op.
        tf.add_to_collection("labels", labels)  # Remember this Op.
        tf.add_to_collection("keep_prob", keep_prob) # Remember this Op.
        
        # Build a Graph that computes predictions from the inference model.
        logits = inference_graph(x, data.train.samples.shape[1], keep_prob)
        
        # save inference graph to file
        tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "inference_regression.pbtxt", as_text=True)
        
        tf.add_to_collection("logits", logits)  # Remember this Op.
        
        # Add to the Graph the Ops that calculate and apply gradients.
        train_op, loss = training_graph(logits, labels, 1e-4)
        
        with tf.name_scope("accuracy"):
            batch_error = tf.abs(tf.subtract(labels, logits))
            err_mean = tf.reduce_mean(tf.cast(batch_error, tf.float32))
            err_min = tf.reduce_min(tf.cast(batch_error, tf.float32))
            err_max = tf.reduce_max(tf.cast(batch_error, tf.float32))
            err_op = tf.reduce_sum(tf.cast(batch_error, tf.float32))
        
        # Add the variable initializer Op.
        init = tf.initialize_all_variables()
        
        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()
        
        # save complete flow graph to file
        tf.train.write_graph(tf.get_default_graph().as_graph_def(),TRAIN_DIR, "complete_regression.pbtxt", as_text=True)

    start_time = time.time()

    # Run training for MAX_STEPS and save checkpoint at the end.
    with tf.Session(graph = graph) as sess:
        # Run the Op to initialize the variables.
        sess.run(init)
        
        # Training loop.
        for step in range(MAX_STEPS):
            # Read a batch of samples and labels.
            samples_batch, labels_batch = data.train.next_batch(BATCH_SIZE)
            # Run one step of the model.
            _, loss_value = sess.run([train_op, loss], feed_dict={x: samples_batch, labels: labels_batch, keep_prob: 0.5})
            
            # Print loss value.
            if step > 0:
                err, err_m, min_err, max_err = sess.run([err_op, err_mean, err_min, err_max], feed_dict={x: data.test.samples, labels: data.test.labels, keep_prob: 1.0})

                # final evaluation and performance metrics
                print('Step %d: loss = %.2f' % (step, loss_value))
                print("##### epoch: %d, %.2f s, error: %f, mean_error: %f, min_err: %f, max_err: %f" % (data.train.epochs_completed, (time.time() - start_time), err, err_m, min_err, max_err))

                # Write a checkpoint.
                saver.save(sess, checkpoint_file, global_step=step)
    
    resout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/data', help='Directory for storing data')
    FLAGS = parser.parse_args()
    tf.app.run()
