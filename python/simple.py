import tensorflow as tf
import numpy as np
from utils.utils import *
import os
from os.path import join, exists


def process_dataset():
    # Import the data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape the data
    NUM_TRAIN = 60000
    NUM_TEST = 10000
    x_train = np.reshape(x_train, (NUM_TRAIN, 28, 28, 1))
    x_test = np.reshape(x_test, (NUM_TEST, 28, 28, 1))
    return x_train, y_train, x_test, y_test


def create_model(train_x, train_y, test_x, test_y, saved_model, filename='model.h5'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=[28, 28, 1]))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_file = join(saved_model, filename)
    if exists(model_file):
        model = tf.keras.models.load_model(model_file)
    else:
        model.fit(train_x, train_y, epochs=5, verbose=1)
    model.evaluate(test_x, test_y)
    return model


def saved_engine(saved_model, engine_path):
    model = tf.keras.models.load_model(saved_model)
    convert_to_frozen(model, engine_path)

    # Evaluate the model on test data

    model = tf.keras.models.load_model(saved_model)


def main():
    x_train, y_train, x_test, y_test = process_dataset()
    saved_model = '/tmp/mnist'
    model = create_model(x_train, y_train, x_test, y_test, saved_model)
    convert_to_frozen(model, '/tmp/mnist/sample')
    data_nums, _, _, _ = x_test.shape
    acc = 0
    batch = 16
    output = inference_with_uff(
        '/tmp/mnist/sample/frozen_graph.uff', x_test[:1][:][:][:], max_batch_size=1)
    print(np.argmax(output[0]))
    # for i in range(batch):
    #     output = inference_with_uff(
    #     '/tmp/mnist/sample/frozen_graph.uff', x_test[i][:][:][:],max_batch_size=1)
    #     print(np.argmax(output[0][i*10:(i+1)*10]),y_test[i])

    # for i in range(data_nums):
    #     input_data = x_test[i][:][:][:]
    #     input_label = y_test[i]
    #     output = inference_with_uff(
    #         '/tmp/mnist/sample/frozen_graph.uff', input_data)
    #     if np.argmax(output[0]) == input_label:
    #         acc += 1
    # print("Engine final acc: {:.4f} %".format(100*acc/data_nums))


if __name__ == '__main__':
    main()
