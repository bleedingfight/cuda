import ctypes
import os

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np
import common
TRT_LOGGER = trt.Logger()

GELU_PLUGIN_LIBRARY = 'build/libgeluplugin.so'
# ctypes.CDLL(GELU_PLUGIN_LIBRARY)

lib = ctypes.cdll.LoadLibrary(GELU_PLUGIN_LIBRARY)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
) as network, trt.UffParser() as parser:
    builder.max_workspace_size = common.GiB(1)
    uff_path = 'models/frozen_graph.uff'
    parser.register_input('x', (1, 28, 28))
    parser.register_output('sequential/Gelu/Gelu/truediv')
    parser.parse(uff_path, network)
    engine = builder.build_cuda_engine(network)

inputs, outputs, bindings, stream = common.allocate_buffers(engine)
context = engine.create_execution_context()
r = common.do_inference(context,
                        bindings=bindings,
                        inputs=inputs,
                        outputs=outputs,
                        stream=stream)
#print([elem.shape for elem in r], np.argmax(r[1]))
print(r)
