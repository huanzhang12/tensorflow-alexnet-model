import tensorflow as tf
from tensorflow.python.platform import gfile
import sys

if len(sys.argv) < 2:
    print("Usage: {} model_file log_dir".format(sys.argv[0]))
    sys.exit(0)

model_filename = sys.argv[1]
LOGDIR = sys.argv[2]

with tf.Session() as sess:
    with gfile.FastGFile(model_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)
        for node in graph_def.node:
            print(node.name)
print("writing summary to", LOGDIR)
train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.close()

