# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:14:53 2016

@author: pooya
"""

'''
A Multilayer Perceptron implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
#import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import csv
from csv_read import MyInput
csvpath = '/home/pooya/Desktop/hpc/new_train.csv'

import tensorflow as tf

train_data_percentage = 60
test_data_percentage = 40

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
# n_input = 784 # MNIST data input (img shape: 28*28)
n_input = 127
# n_classes = 10 # MNIST total classes (0-9 digits)
n_classes = 2

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
    return tf.matmul(layer_2, _weights['out']) + _biases['out']

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    count = 0
    with open(csvpath, 'rU') as count_file:
        csv_reader = csv.reader(count_file)
        for row in csv_reader:
            count += 1
    for epoch in range(training_epochs):
        avg_cost = 0.
        #total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = int(count/batch_size)
        total_train_batch = int(total_batch * train_data_percentage/100)
        total_test_batch = total_batch - total_train_batch
        # Loop over all batches
        for i in range(total_train_batch):
            #batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs, batch_ys = MyInput(csvpath, batch_size, i)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization Finished!"

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    indicator = int(total_batch/total_test_batch) - 1
    test_batch_xs, test_batch_ys = MyInput(csvpath, total_test_batch, indicator)
    print "Accuracy:", accuracy.eval({x: test_batch_xs, y: test_batch_ys})
