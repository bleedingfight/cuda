import tensorflow as tf
import numpy as np
import os
from os.path import exists
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from functools import partial
import datetime
from utils.utils import load_data
WORKING_DIR = os.environ.get("TRT_WORKING_DIR") or os.path.dirname(
    os.path.realpath(__file__))

MODEL_DIR = os.path.join(WORKING_DIR, 'models')


def build_model():
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.InputLayer(input_shape=[1, 28, 28], name="InputLayer"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32))
    model.add(tf.keras.layers.Activation(
        activation=tf.nn.gelu, name="Gelu"))
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


def save_model(model, graph_def_path='/models'):
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
                          name="frozen_graph.pb",
                          as_text=False)


if __name__ == "__main__":
    model = train_model('/tmp/mnist/model.h5')
    graphdef_path = '/tmp/mnist'
    save_model(model, graph_def_path=graphdef_path)
