# Basic Operations example using TensorFlow library


from __future__ import print_function
import tensorflow as tf


# Basic constant operations
a = tf.constant(2)
b = tf.constant(3)

# Launch the default graph
with tf.Session() as sess:
    print("a=2, b=3")
    print("Addition with constants: %i " % sess.run(a+b))
    print("Multiplication with constants: %i " % sess.run(a*b))

# Basic Operations
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)


# Define some operations
add = tf.add(a,b)
mul = tf.multiply(a, b)

# lunch the default graph
with tf.Session() as sess:
    print("Addition with variables: %i" % sess.run(add, feed_dict={a: 2, b: 3}))
    print("Multiplication with variables: %i" % sess.run(mul, feed_dict={a: 2, b: 3}))

    # more details

    matrix1 = tf.constant([[3., 3.]])
    matrix2 = tf.constant([[2.],[2.]])


    product = tf.matmul(matrix1, matrix2)


    with tf.Session() as sess:
        result = sess.run(product)
        print(result)
