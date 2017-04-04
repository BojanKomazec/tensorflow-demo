# mon.csv contains blood pressure (sys and dys) and heart rate measurements.
# sys and dys have linear relation so we'll use these two columns.

from __future__ import print_function

import tensorflow as tf
import csv
import numpy
import matplotlib.pyplot as plt

def main():
    rng = numpy.random

    # Different parameters for learning
    learning_rate = 0.01
    training_epochs = 1000
    display_step = 100

    # Load datat from the file
    tuples_list = []

    with open('mon.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        is_header_row = True
        for row in reader:
            if is_header_row:
                is_header_row = False
                continue
            tuples_list.append((int(row[0]), int(row[1])))

    # we want to use first 2/3 of all examples as training set and next 1/3 as validation set
    cut_off_index = int(round((2/3)*len(tuples_list)))
    training_set_list = tuples_list[:cut_off_index]
    test_set_list = tuples_list[cut_off_index + 1:]

    # Training Data
    train_x = numpy.asarray([t[0] for t in training_set_list])
    train_y = numpy.asarray([t[1] for t in training_set_list])
    n_samples = train_x.shape[0]

    plt.plot(train_x, train_y, 'bo', label='Blood pressure examples (Training data)')
    plt.xlabel('sys (systolic)')
    plt.ylabel('dys (diastolic)')
    plt.legend()
    plt.show()

    # Create placeholder for providing inputs
    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    # create weights and bias and initialize with random number
    W = tf.Variable(rng.randn(), name="weight")
    b = tf.Variable(rng.randn(), name="bias")

    # Construct a linear model using Y=WX+b
    pred = tf.add(tf.multiply(X, W), b)

    # Calculate Mean squared error
    cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

    # Gradient descent to minimize mean sequare error
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        print("Training started")

        # Fit all training data
        for epoch in range(training_epochs):
            for (x, y) in zip(train_x, train_y):
                #create small batch of training and testing data and feed it to model
                sess.run(optimizer, feed_dict={X: x, Y: y})

            # Display training information after each N step
            if (epoch+1) % display_step == 0:
                c = sess.run(cost, feed_dict={X: train_x, Y:train_y})
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                    "W=", sess.run(W), "b=", sess.run(b))

        print("Training completed")

        training_cost = sess.run(cost, feed_dict={X: train_x, Y: train_y})
        print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

        # Plot line with fitted data
        plt.plot(train_x, train_y, 'ro', label='Original data')
        plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

        # Testing
        print("Testing started")
        test_x = numpy.asarray([t[0] for t in test_set_list])
        test_y = numpy.asarray([t[1] for t in test_set_list])

        #Calculate Mean square error
        print("Calculate Mean square error")
        testing_cost = sess.run(
            tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_x.shape[0]),
            feed_dict={X: test_x, Y: test_y})  # same function as cost above
        print("Testing cost=", testing_cost)
        print("Absolute mean square loss difference:", abs(
            training_cost - testing_cost))

        plt.plot(test_x, test_y, 'bo', label='Testing data')
        plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Fitted line')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()
