import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorrt as trt
from utils.common import *
from os.path import join, basename, dirname


def load_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = np.reshape(x_train, (-1, 1, 28, 28))
    x_test = np.reshape(x_test, (-1, 1, 28, 28))
    return x_train, y_train, x_test, y_test


def convert_to_frozen(model, graph_def_path='/tmp/models', filename="frozen_graph.pb"):
    """转换训练好的tensorflow模型为frozen_graph.pb

    Args:
        model (Module): 加载完成的tensorflow模型
        graph_def_path (str, optional): 保存graphdef文件的位置. Defaults to '/tmp/models'.
    """
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        x=tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    layers = [op.name for op in frozen_func.graph.get_operations()]
    for layer in layers:
        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=graph_def_path,
                          name="filename",
                          as_text=False)


def inference_with_uff(uff_model_file, input_data, max_batch_size=16, saved_engine=True):

    TRT_LOGGER = trt.Logger()
    input_data = input_data.ravel()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
    ) as network, trt.UffParser() as parser:
        parser.register_input('x', (1, 28, 28))
        parser.register_output('Identity')
        parser.parse(uff_model_file, network)
        builder.max_batch_size = max_batch_size
        config = builder.create_builder_config()
        config.max_workspace_size = MiB(10)
        # config.max_batch_size = 10
        engine = builder.build_engine(network, config)
        if saved_engine:
            with open(join(dirname(uff_model_file), 'model.engine'), 'wb') as f:
                f.write(engine.serialize())

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    np.copyto(inputs[0].host, input_data)
    # print("====>",np.sum(inputs[0].host-input_data))

    context = engine.create_execution_context()
    r = do_inference(context,
                     bindings=bindings,
                     inputs=inputs,
                     outputs=outputs,
                     stream=stream)
    return r


def inference_with_engine(engine, input_data):
    assert isinstance(engine, str) or isinstance(
        trt.tensorrt.ICudaEngine), "engine must be file or deserialized engine"
    if isinstance(engine, 'str'):
        with open(engine, 'rb') as f:
            with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
                engine = runtime.deserialize_cuda_engine(f.read())
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    # def load_test_case(pagelocked_buffer):
    # _, _, x_test, y_test = load_data()
    # num_test = len(x_test)
    # np.copyto(pagelocked_buffer, img)
    # return y_test[case_num]
    context = engine.create_execution_context()
    r = do_inference(context,
                     bindings=bindings,
                     inputs=inputs,
                     outputs=outputs,
                     stream=stream)
    return r
