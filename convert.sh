#!/bin/bash
saved_model=/usr/local/TensorRT-7.1.3.4/samples/python/uff_custom_plugin/saved_model/mnist
tensorrt_model=/tmp/saved_model_trt
saved_model_cli convert --dir ${saved_model} --tag_set serve --output_dir ${tensorrt_model} tensorrt
