import tensorflow as tf
import sys


def main(argv):
    # Parse command
    task = int(argv[1])
    cluster_spec = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
    server = tf.train.Server(cluster_spec, job_name="local", task_index=task)

    print("Initialising server {}".format(task))
    server.start()

    # This server will now wait for instructions - n.b. that we did not
    # define an interrupt signal, so you have to close the terminal to kill it
    server.join()


if __name__ == '__main__':
    main(sys.argv)