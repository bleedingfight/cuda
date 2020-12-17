import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
uff_file = '/usr/local/TensorRT-7.1.3.4/samples/python/uff_custom_plugin/models/frozen_graph.uff'
input_name = "x"
output_name = "Identity"


def convert_uff_to_engine(model_file):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
    ) as network, trt.UffParser() as parser:
        parser.register_input(input_name, (1, 28, 28))
        parser.register_output(output_name)
        parser.parse(model_file, network)
        engine = builder.build_cuda_engine(network)
        serialized_engine = engine.serialize()
        with open("models/sample.engine", "wb") as f:
            f.write(engine.serialize())


convert_uff_to_engine(uff_file)
