import tensorflow.compat.v1 as tf
from utils import load_data
import numpy as np

model_path = 'models/frozen_graph.pb'
graph_def = tf.GraphDef()
with tf.gfile.GFile(model_path, 'rb') as f:
    graph_def.ParseFromString(f.read())

nodes = [node.name for node in graph_def.node]
_, _, test_x, test_y = load_data()
input_name = 'x:0'
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    erf_node = sess.graph.get_tensor_by_name('sequential/Gelu/Gelu/truediv:0')
    input_node = sess.graph.get_tensor_by_name(input_name)
    # test_data = test_x[0][:][:][:][np.newaxis, :, :, :]
    test_data = np.zeros(shape=(
        1,
        1,
        28,
        28,
    ))
    test_label = test_y[0]
    erf_in = sess.run([erf_node], feed_dict={'x:0': test_data})
    # print("erf value:{},output_label:{}".format(erf_out[0].shape, test_label))
    print(erf_in)
