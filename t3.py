# k-nearest neighbor

from __future__ import print_function

import numpy as np
import tensorflow as tf

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/lino/googletensorflow/",one_hot=True)


# we limit the data
X_train, Y_train = mnist.train.next_batch(5000)
X_test, Y_test = mnist.test.next_batch(200)


#tf graph input

x_train = tf.placeholder("float", [None, 784])
x_test = tf.placeholder("float", [784])


# Nearest Neighbor Calculation

distance = tf.reduce_sum(tf.abs(tf.add(x_train, tf.negative(x_test))), reduction_indices=1)

# Prediction
pred = tf.arg_min(distance, 0)

accuracy = 0

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(X_test)):
        # get nearest neighbor
        nn_index = sess.run(pred, feed_dict={x_train: X_train, x_test: X_test[i, :]})

        # get nearets neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Y_train[nn_index]), \
                                                  "True class:", np.argmax(Y_test[i]))
        # calculate accuracy
        if np.argmax(Y_train[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)

        print("Done")
        print("Accuracy:", accuracy)
        
