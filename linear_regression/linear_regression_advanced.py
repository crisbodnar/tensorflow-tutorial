import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# Generate samples of a function we are trying to predict:
samples = 100
n_dim = 1
xs = np.linspace(-5, 5, samples)
# We will attempt to fit this function
ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, samples)

# First, create TensorFlow placeholders for input data (xs) and
# output (ys) data. Placeholders are inputs to the computation graph.
# When we run the graph, we need to feed values for the placerholders into the graph.
X = tf.placeholder(tf.float32, shape=(None,), name='x')
Y = tf.placeholder(tf.float32, shape=(None,), name='y')

# We will try minimzing the mean squared error between our predictions and the
# output. Our predictions will take the form X*W + b, where X is input data,
# W are ou weights, and b is a bias term:
# minimize ||(X*w + b) - y||^2
# To do so, you will need to create some variables for W and b. Variables
# need to be initialised; often a normal distribution is used for this.
W = tf.Variable(0.0)
b = tf.Variable(1.0)

# Next, you need to create a node in the graph combining the variables to predict
# the output: Y = X * w + b. Find the appropriate TensorFlow operations to do so.
predictions = X * W + b


# Finally, we need to define a loss that can be minimized using gradient descent:
# The loss should be the mean squared difference between predictions
# and outputs.
loss = tf.reduce_mean(tf.square(Y - predictions))

# Use gradient descent to optimize your variables
learning_rate = 0.001
optimize_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# We create a session to use the graph and initialize all variables
session = tf.Session()
session.run(tf.global_variables_initializer())

# Optimisation loop
epochs = 1000
delta = 0.000001

epoch = tf.constant(0.0)
previous_cost = tf.constant(0.0)
cost = tf.constant(session.run(loss, feed_dict={X: xs, Y: ys}))


def condition(epoch, previous_cost, cost):
    return tf.logical_and(tf.less(epoch, epochs), tf.greater(tf.abs(previous_cost - cost), delta))


def body(epoch, _, cost):
    _, new_cost = session.run([optimize_op, loss], feed_dict={X: xs, Y: ys})
    return epoch + 1, cost, new_cost


tf.while_loop(condition, body, (epoch, previous_cost, cost))

predictions = session.run(predictions, feed_dict={X: xs, Y: ys})
plt.plot(xs, predictions)
plt.plot(xs, ys)
plt.show()
