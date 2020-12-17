import tensorrt as trt
import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def allocate_buffers(engine, batch_size, data_type):
    """
   This is the function to allocate buffers for input and output in the device
   Args:
      engine : The path to the TensorRT engine. 
      batch_size : The batch size for execution time.
      data_type: The type of the data for input and output, for example trt.float32. 
   
   Output:
      h_input_1: Input in the host.
      d_input_1: Input in the device. 
      h_output_1: Output in the host. 
      d_output_1: Output in the device. 
      stream: CUDA stream.

   """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size *
                                      trt.volume(engine.get_binding_shape(0)),
                                      dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size *
                                     trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


def do_inference(engine, pics_1, h_input_1, d_input_1, h_output, d_output,
                 stream, batch_size, height, width):
    """
   This is the function to run the inference
   Args:
      engine : Path to the TensorRT engine 
      pics_1 : Input images to the model.  
      h_input_1: Input in the host         
      d_input_1: Input in the device 
      h_output_1: Output in the host 
      d_output_1: Output in the device 
      stream: CUDA stream
      batch_size : Batch size for execution time
      height: Height of the output image
      width: Width of the output image
   
   Output:
      The list of output images

   """

    load_images_to_buffer(pics_1, h_input_1)

    with engine.create_execution_context() as context:
        # Transfer input data to the GPU.
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)

        # Run inference.

        context.profiler = trt.Profiler()
        context.execute(batch_size=1, bindings=[int(d_input_1), int(d_output)])

        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        # Synchronize the stream
        stream.synchronize()
        # Return the host output.
        out = h_output.reshape((batch_size, -1, height, width))
        return out


def build_engine(onnx_path, shape=[1, 229, 229, 3]):
    """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file. 
      shape : Shape of the input of the ONNX file. 
  """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def main():
    onnx_path = '/home/biomind/custom_plugin/models.onnx'
    engine_path = 'inception_model.engine'
    engine = build_engine(onnx_path)
    #save_engine(engine, engine_path)
    #engine_plan = load_engine(plan_path=engine_path)
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)),
                                    dtype=trt.nptype(trt.float32))

    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)),
                                     dtype=trt.nptype(trt.float32))

    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    with engine.create_execution_context() as context:
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async(bindings=[int(d_input),
                                        int(d_output)],
                              stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        print(h_output)


if __name__ == "__main__":
    main()
