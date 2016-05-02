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
csvpath = '/home/pooya/Desktop/hpc/test/new_train_2.csv'

import tensorflow as tf
def run_experiments(n_layers):
    train_data_percentage = 60
    test_data_percentage = 40

    # Parameters
    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100
    display_step = 1

    # Network Parameters
    n_input = 127
    n_classes = 2
    n_hidden = []
    for i in range(n_layers):
        n_hidden.append(256)
    n_hidden[0] = n_input
    n_hidden.append(n_classes)

    # Store layers weight & bias
    weights = {}
    for i in range(n_layers):
        #print "(%d, %d)" % (n_hidden[i],n_hidden[i+1])
        weights[i] = tf.Variable(tf.random_normal([n_hidden[i], n_hidden[i+1]]))

    biases = {}
    for i in range(n_layers):
        biases[i] = tf.Variable(tf.random_normal([n_hidden[i+1]]))



    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    # Create model
    def multilayer_perceptron(_X, _weights, _biases):
        layers = []
        layers.append(tf.nn.relu(tf.add(tf.matmul(_X, _weights[0]), _biases[0]))) #Hidden layer with RELU activation
        for i in range(1,n_layers-1):
            layers.append(tf.nn.relu(tf.add(tf.matmul(layers[i-1], _weights[i]), _biases[i])))
            #print "layers [%d] =  (layers %d, weights %d + b %d)" % (i,i-1,i,i)
        return tf.matmul(layers[n_layers-2], _weights[n_layers-1]) + _biases[n_layers-1]

    # Construct model
    pred = multilayer_perceptron(x, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    correct_prediction_train = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    train_accuracy = tf.reduce_mean(tf.cast(correct_prediction_train, "float"))

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

                #print('\ntrain batch: %d'%i) # To see which batch is loaded correctly

                # Fit training using batch data
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                # Compute average loss
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch

                correct_prediction_train_ratio = sess.run(train_accuracy, feed_dict={x: batch_xs, y: batch_ys})
                print "\nEpoch:", '%d' %(epoch+1), '/', '%d\n' %training_epochs , "Iteration:", '%d' %(i+1), '/', '%d\n'%total_train_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print "\nEpoch:", '%d' %(epoch+1), '/', '%d\n' %training_epochs , "Iteration:", '%d' %(i+1), '/', '%d\n'%total_train_batch, "cost=", "{:.9f}".format(avg_cost)
                print "\nCorrect prediction ratio for train data in each batch:", '%f'%correct_prediction_train_ratio
        print "Optimization Finished!"

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        indicator = int(total_batch/total_test_batch) - 1
        test_batch_xs, test_batch_ys = MyInput(csvpath, total_test_batch, indicator)
        print "Accuracy:", accuracy.eval({x: test_batch_xs, y: test_batch_ys})

if __name__ == '__main__':
    for n_layers in range(3,200,10):
        run_experiments(n_layers)