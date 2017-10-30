# Linear regression


from __future__ import print_function

import tensorflow as tf
import numpy
import matplotlib.pyplot as plt



rng = numpy.random
#parameters
learning_rate = 0.01
epochs = 1000000
display_step = 500


# Training Data
x_train = numpy.asarray([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9])
y_train = numpy.asarray([10,20,30,40,50,60,70,80,90])


n_samples = x_train.shape[0]


X = tf.placeholder("float")
Y = tf.placeholder("float")


# set a model weights

w = tf.Variable(rng.randn(), name="weight")
b = tf.Variable(rng.randn(), name="bias")

# construct a linear model
pred = tf.add(tf.multiply(X, w), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

# gradient descnet
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
    sess.run(init)

    # fit all training data
    for epoch in range(epochs):
        for (x, y) in zip(x_train, y_train):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # display logs por epochs step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: x_train, Y: y_train})
            print("Epoch", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                  "w=", sess.run(w), "b=", sess.run(b))

    print("Optimization Finished")
    training_cost = sess.run(cost, feed_dict={X: x_train, Y: y_train})
    print("training cost=", training_cost, "w=", sess.run(w), "b=", sess.run(b), '\n')


    # graphic display
    plt.plot(x_train, y_train, 'ro', label='Original data')
    plt.plot(x_train, sess.run(w) * x_train + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()


    # testing examples
    x_test = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
    y_test = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

    print("testing...(Mean square loss Comparison)")
    testing_cost = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * x_test.shape[0]),
        feed_dict={X: x_test, Y: y_test})
    print("testing cost=", testing_cost)
    print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

    plt.plot(x_test, y_test, 'bo', label='tetsing data')
    plt.plot(x_train, sess.run(w) * x_train * sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
