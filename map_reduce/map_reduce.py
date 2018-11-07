import numpy as np
import tensorflow as tf

# TODO create a cluster spec matching your server spec

task_input = tf.placeholder(tf.float32, 100)

# First part: compute mean of half of the input data
with tf.device("/job:local/task:0"):
    local_input = tf.slice(task_input, [50], [-1])
    local_mean = tf.reduce_mean(local_input)

# TODO do another half of the computation using another device
# TODO compute the overall result by combining both results



# TODO Fill in the session specification
with tf.Session() as sess:
    # Sample some data for the computation
    data = np.random.random(100)

    # TODO run the session to compute the overall using your workers
    # and the input data. Output the result.
