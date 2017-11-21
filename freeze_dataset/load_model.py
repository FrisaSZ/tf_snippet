import os
import numpy as np
import tensorflow as tf
from create_model import USE_DATASET

import sys
print(sys.version)


def load_graph(model_file):
    #_ = tf.contrib.data.Dataset
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


model_file = os.path.join(os.getcwd(), 'model/frozen_model.pb')
x_placeholder = 'import/x'
make_iterator = 'import/MakeIterator'
output_layer = 'import/y'

graph = load_graph(model_file)
op_list = graph.get_operations()
x_op = graph.get_operation_by_name(x_placeholder)
make_iterator_op = graph.get_operation_by_name(make_iterator)
output = graph.get_operation_by_name(output_layer)
feed_dict = {x_op.outputs[0]: [[1, 2, 3, 4]]}
with tf.Session(graph=graph) as sess:
    sess.run(make_iterator_op, feed_dict=feed_dict)
    if USE_DATASET:
        results = sess.run(output.outputs[0])
    else:
        results = sess.run(output.outputs[0], feed_dict=feed_dict)
    print(results)
