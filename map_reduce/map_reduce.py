import numpy as np
import tensorflow as tf

cluster_spec = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
task_input = tf.placeholder(tf.float32, 100)

# First part: compute mean of half of the input data
with tf.device("/job:local/task:0"):
    local_input_task0 = tf.slice(task_input, [50], [-1])
    local_mean_task0 = tf.reduce_mean(local_input_task0)

# TODO do another half of the computation using another device
with tf.device("/job:local/task:1"):
    local_input_task1 = tf.slice(task_input, [0], [50])
    local_mean_task1 = tf.reduce_mean(local_input_task1)

# TODO compute the overall result by combining both results
global_mean = (local_mean_task0 + local_mean_task1) / 2.0

# TODO Fill in the session specification
with tf.Session("grpc://localhost:2222") as sess:
    # Sample some data for the computation
    data = np.random.random(100)

    # and the input data. Output the result.
    mean = sess.run(global_mean, feed_dict={task_input: data})
    print(mean)