#
# Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

# This file contains functions for training a TensorFlow model
import tensorflow as tf
import numpy as np
import os
from os.path import exists
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from functools import partial
import datetime
from utils import load_data
WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(
    os.path.realpath(__file__))

MODEL_DIR = os.path.join(WORKING_DIR, 'models')


def build_model():
    # current_av = partial(tf.nn.gelu, approximate=False)
    # Create the keras model
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.InputLayer(input_shape=[1, 28, 28], name="InputLayer"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation(activation=tf.nn.gelu, name="Gelu"))
    model.add(
        tf.keras.layers.Dense(10, activation=tf.nn.softmax,
                              name="OutputLayer"))
    return model


def train_model(model_name, logs_path='/tmp/mnist_summary/'):
    # Build and compile model
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Load data
    x_train, y_train, x_test, y_test = load_data()
    callback = tf.keras.callbacks.TensorBoard(
        log_dir="{}_{}".format(logs_path,
                               datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
    if not exists(model_name):
        # Train the model on the data
        model.fit(x_train,
                  y_train,
                  epochs=10,
                  validation_data=(x_test, y_test),
                  verbose=1,
                  callbacks=[callback])

        # Evaluate the model on test data
        model.save(model_name, datetime.datetime.now().strftime)
    model = tf.keras.models.load_model(model_name)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test loss: {}\nTest accuracy: {}".format(test_loss, test_acc))

    return model


def save_model(model, pb_file='models'):
    saved_model = "saved_model/mnist"
    model.save(saved_model)
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
                          logdir=pb_file,
                          name="frozen_graph.pb",
                          as_text=False)


if __name__ == "__main__":
    model = train_model('/tmp/mnist/model.h5')
    save_model(model)
