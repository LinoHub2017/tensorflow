# Convolutional Neural Network

from __future__ import print_function
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/lino/googletensorflow/,one_hot=True")

# Parameters
learning_rate = 0.001
epochs = 10
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784
n_classes = 10
dropout = 0.75

x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])
keep_prob = tf.placeholder('float')

# create some wrappers for simplicity
def conv2d(x, w, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x,w, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    #MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# create a model
def conv_net(x, weights, biases, dropout):
    #reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # convolution layer 1
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max pooling
    conv1 = maxpool2d(conv1, k=2)

    # convolution layer 2
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    #Max Pooling
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input

    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.relu(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


print("*********************")

# Store layers weight and bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out':tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))

}
print("&&&&&&&&&&&&&&&&&&&&&&&")
# construct model
pred = conv_net(x, weights ,biases, keep_prob)

# Define loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=prediction, logits=y))
#optimizer = tf.train.AdamOptimizer().minimize(cost)
optmizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

print("MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM")
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # keep training untill reach max iteractions
    while step * batch-size < epochs:
        batch_x, batch_y = mnist.train.next_batch(batch-size)
        # Run optimization op(backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: bacth_y, keep_prob: dropout})

        if step % display_step == 0:
            #Calculate batch loss and accuarcy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iteration" + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished")
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))
